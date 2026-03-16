"""
LSTM Training Pipeline Script

This script demonstrates end-to-end training workflow for the LSTM Transfer Value Predictor:
1. Loads the training dataset
2. Prepares sequences using SequencePreparator
3. Trains the LSTM model with early stopping and model checkpointing
4. Saves the trained model
5. Generates training reports and visualizations

Usage:
    python scripts/train_lstm.py [--config CONFIG_PATH]
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.lstm_model import LSTMTransferValuePredictor
from src.models.lstm_utils import SequencePreparator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_training_data(data_path: str) -> pd.DataFrame:
    """
    Load training dataset from CSV
    
    Args:
        data_path: Path to training dataset CSV file
    
    Returns:
        DataFrame with training data
    """
    logger.info(f"Loading training data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    return df


def prepare_temporal_data(df: pd.DataFrame, n_timesteps: int = 12) -> pd.DataFrame:
    """
    Prepare temporal data for sequence generation
    
    Since the dataset has only one snapshot per player, we'll create
    synthetic temporal sequences by generating historical data points
    with realistic variations.
    
    Args:
        df: Input DataFrame
        n_timesteps: Number of time steps to generate per player
    
    Returns:
        DataFrame with temporal ordering
    """
    logger.info(f"Preparing temporal data with {n_timesteps} timesteps per player")
    
    # Ensure player_id exists
    if 'player_id' not in df.columns:
        raise ValueError("Dataset must contain 'player_id' column")
    
    temporal_data = []
    
    # For each player, create a time series
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"Processing player {idx}/{len(df)}")
        
        player_id = row['player_id']
        
        # Generate time series for this player
        for t in range(n_timesteps):
            # Create a copy of the row
            new_row = row.copy()
            
            # Add temporal ordering
            new_row['date'] = pd.Timestamp('2020-01-01') + pd.Timedelta(weeks=t)
            new_row['timestep'] = t
            
            # Add realistic variations to numerical features
            # Earlier timesteps have more variation (representing historical uncertainty)
            variation_factor = 1.0 - (t / n_timesteps) * 0.3  # 0.7 to 1.0
            
            # Apply variations to performance metrics
            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numerical_cols:
                if col not in ['player_id', 'market_value', 'fee']:
                    if pd.notna(new_row[col]):
                        # Add random variation (±10% scaled by variation_factor)
                        noise = np.random.normal(0, 0.1) * variation_factor
                        new_row[col] = new_row[col] * (1 + noise)
                        
                        # Ensure non-negative for certain metrics
                        if col in ['age', 'Appearances', 'Minutes', 'Goals', 'Assists']:
                            new_row[col] = max(0, new_row[col])
            
            # Market value should show a trend (increasing towards current)
            if pd.notna(new_row['market_value']):
                # Earlier values are lower, with some noise
                trend_factor = 0.6 + (t / n_timesteps) * 0.4  # 0.6 to 1.0
                noise = np.random.normal(0, 0.05)
                new_row['market_value'] = new_row['market_value'] * trend_factor * (1 + noise)
                new_row['market_value'] = max(0, new_row['market_value'])
            
            temporal_data.append(new_row)
    
    # Create DataFrame
    temporal_df = pd.DataFrame(temporal_data)
    
    # Sort by player and date
    temporal_df = temporal_df.sort_values(['player_id', 'date']).reset_index(drop=True)
    
    logger.info(f"Created temporal dataset with {len(temporal_df)} records "
               f"({len(df)} players × {n_timesteps} timesteps)")
    
    return temporal_df


def select_features(df: pd.DataFrame) -> list:
    """
    Select relevant features for LSTM training
    
    Args:
        df: Input DataFrame
    
    Returns:
        List of feature column names
    """
    # Exclude non-feature columns
    exclude_columns = [
        'Player Name', 'player_id', 'date', 'Club', 
        'market_value', 'fee'  # fee is target, market_value is related
    ]
    
    # Select numerical features
    feature_columns = [
        col for col in df.columns 
        if col not in exclude_columns and df[col].dtype in ['int64', 'float64']
    ]
    
    logger.info(f"Selected {len(feature_columns)} features for training")
    logger.info(f"Features: {feature_columns[:10]}...")  # Show first 10
    
    return feature_columns


def prepare_sequences_for_training(
    df: pd.DataFrame,
    feature_columns: list,
    target_column: str = 'market_value',
    sequence_length: int = 10,
    prediction_horizons: list = None
) -> tuple:
    """
    Prepare sequences for LSTM training
    
    Args:
        df: DataFrame with temporal data
        feature_columns: List of feature column names
        target_column: Target column name
        sequence_length: Length of input sequences
        prediction_horizons: List of prediction horizons
    
    Returns:
        Tuple of (X, y, player_ids, splits)
    """
    if prediction_horizons is None:
        prediction_horizons = [1, 3, 6]
    
    logger.info(f"Preparing sequences with length={sequence_length}, horizons={prediction_horizons}")
    
    # Initialize sequence preparator
    config = {
        'sequence_length': sequence_length,
        'prediction_horizons': prediction_horizons,
        'min_sequence_length': 5
    }
    preparator = SequencePreparator(config)
    
    # Prepare sequences
    X, y, player_ids = preparator.prepare_sequences(
        df=df,
        feature_columns=feature_columns,
        target_column=target_column,
        player_id_column='player_id',
        date_column='date'
    )
    
    # Split into train/val/test
    splits = preparator.split_sequences(
        X=X,
        y=y,
        player_ids=player_ids,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    return X, y, player_ids, splits


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict = None
) -> tuple:
    """
    Train LSTM model with early stopping and checkpointing
    
    Args:
        X_train: Training sequences
        y_train: Training targets
        X_val: Validation sequences
        y_val: Validation targets
        config: Model configuration
    
    Returns:
        Tuple of (model, history)
    """
    if config is None:
        config = {
            'sequence_length': X_train.shape[1],
            'encoder_units': [128, 64],
            'decoder_units': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'prediction_horizons': [1, 3, 6]
        }
    
    logger.info("Initializing LSTM model")
    model = LSTMTransferValuePredictor(config)
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model.build_model(input_shape)
    
    logger.info("Model architecture:")
    logger.info(model.get_model_summary())
    
    # Train model
    logger.info("Starting training with backpropagation through time")
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=100,
        batch_size=32,
        early_stopping_patience=15,
        reduce_lr_patience=7,
        verbose=1
    )
    
    logger.info("Training completed successfully")
    
    return model, history


def evaluate_model(
    model: LSTMTransferValuePredictor,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Evaluate trained model on test set
    
    Args:
        model: Trained LSTM model
        X_test: Test sequences
        y_test: Test targets
    
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating model on test set")
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test, verbose=1)
    
    # Get predictions for detailed analysis
    predictions = model.predict(X_test)
    
    # Calculate additional metrics per horizon
    horizon_metrics = {}
    for i, horizon in enumerate(model.prediction_horizons):
        y_true = y_test[:, i]
        y_pred = predictions[:, i]
        
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        horizon_metrics[f'{horizon}_month'] = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
    
    metrics['horizon_metrics'] = horizon_metrics
    
    logger.info("Evaluation metrics:")
    logger.info(f"  Overall - Loss: {metrics['loss']:.4f}, MAE: {metrics['mae']:.4f}, MAPE: {metrics['mape']:.2f}%")
    for horizon, hmetrics in horizon_metrics.items():
        logger.info(f"  {horizon} - RMSE: {hmetrics['rmse']:.4f}, MAE: {hmetrics['mae']:.4f}, MAPE: {hmetrics['mape']:.2f}%")
    
    return metrics


def save_model_and_artifacts(
    model: LSTMTransferValuePredictor,
    history: dict,
    metrics: dict,
    output_dir: str = 'models/lstm'
) -> None:
    """
    Save trained model, training history, and evaluation metrics
    
    Args:
        model: Trained LSTM model
        history: Training history
        metrics: Evaluation metrics
        output_dir: Output directory for saving artifacts
    """
    logger.info(f"Saving model and artifacts to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for versioning
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_path = os.path.join(output_dir, f'lstm_model_{timestamp}.keras')
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save latest model (overwrite)
    latest_model_path = os.path.join(output_dir, 'lstm_model_latest.keras')
    model.save_model(latest_model_path)
    logger.info(f"Latest model saved to {latest_model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, f'training_history_{timestamp}.json')
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        history_serializable = {
            k: [float(v) for v in vals] for k, vals in history.items()
        }
        json.dump(history_serializable, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    # Save evaluation metrics
    metrics_path = os.path.join(output_dir, f'evaluation_metrics_{timestamp}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation metrics saved to {metrics_path}")
    
    # Save model configuration
    config_path = os.path.join(output_dir, f'model_config_{timestamp}.json')
    with open(config_path, 'w') as f:
        json.dump(model.get_config(), f, indent=2)
    logger.info(f"Model configuration saved to {config_path}")


def generate_training_visualizations(
    history: dict,
    output_dir: str = 'models/lstm'
) -> None:
    """
    Generate training visualizations
    
    Args:
        history: Training history dictionary
        output_dir: Output directory for saving plots
    """
    logger.info("Generating training visualizations")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LSTM Training Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Training and Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(history['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MAE
    ax2 = axes[0, 1]
    ax2.plot(history['mae'], label='Training MAE', linewidth=2)
    ax2.plot(history['val_mae'], label='Validation MAE', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.set_title('Mean Absolute Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: MAPE
    ax3 = axes[1, 0]
    ax3.plot(history['mape'], label='Training MAPE', linewidth=2)
    ax3.plot(history['val_mape'], label='Validation MAPE', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('MAPE (%)')
    ax3.set_title('Mean Absolute Percentage Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate (if available)
    ax4 = axes[1, 1]
    if 'lr' in history:
        ax4.plot(history['lr'], linewidth=2, color='green')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
    else:
        # Show summary statistics instead
        final_train_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        best_val_loss = min(history['val_loss'])
        epochs_trained = len(history['loss'])
        
        summary_text = f"""
        Training Summary:
        
        Epochs Trained: {epochs_trained}
        
        Final Training Loss: {final_train_loss:.4f}
        Final Validation Loss: {final_val_loss:.4f}
        Best Validation Loss: {best_val_loss:.4f}
        
        Final Training MAE: {history['mae'][-1]:.4f}
        Final Validation MAE: {history['val_mae'][-1]:.4f}
        
        Final Training MAPE: {history['mape'][-1]:.2f}%
        Final Validation MAPE: {history['val_mape'][-1]:.2f}%
        """
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='center', family='monospace')
        ax4.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_dir, f'training_curves_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Training visualization saved to {plot_path}")
    
    plt.close()


def generate_training_report(
    metrics: dict,
    history: dict,
    output_dir: str = 'models/lstm'
) -> None:
    """
    Generate comprehensive training report
    
    Args:
        metrics: Evaluation metrics
        history: Training history
        output_dir: Output directory
    """
    logger.info("Generating training report")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    report_path = os.path.join(output_dir, f'training_report_{timestamp}.md')
    
    with open(report_path, 'w') as f:
        f.write("# LSTM Transfer Value Predictor - Training Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Architecture\n\n")
        f.write("- **Type:** Encoder-Decoder LSTM\n")
        f.write("- **Encoder:** 2 LSTM layers (128, 64 units) with dropout (0.2)\n")
        f.write("- **Decoder:** 2 LSTM layers (64, 32 units) with dropout (0.2)\n")
        f.write("- **Optimizer:** Adam (learning_rate=0.001)\n")
        f.write("- **Loss Function:** MSE (Mean Squared Error)\n")
        f.write("- **Metrics:** MAE, MAPE\n\n")
        
        f.write("## Training Configuration\n\n")
        f.write(f"- **Epochs Trained:** {len(history['loss'])}\n")
        f.write("- **Batch Size:** 32\n")
        f.write("- **Early Stopping Patience:** 15 epochs\n")
        f.write("- **Learning Rate Reduction Patience:** 7 epochs\n")
        f.write("- **Prediction Horizons:** 1-month, 3-month, 6-month\n\n")
        
        f.write("## Training Results\n\n")
        f.write("### Overall Metrics\n\n")
        f.write(f"- **Final Training Loss:** {history['loss'][-1]:.4f}\n")
        f.write(f"- **Final Validation Loss:** {history['val_loss'][-1]:.4f}\n")
        f.write(f"- **Best Validation Loss:** {min(history['val_loss']):.4f}\n")
        f.write(f"- **Final Training MAE:** {history['mae'][-1]:.4f}\n")
        f.write(f"- **Final Validation MAE:** {history['val_mae'][-1]:.4f}\n")
        f.write(f"- **Final Training MAPE:** {history['mape'][-1]:.2f}%\n")
        f.write(f"- **Final Validation MAPE:** {history['val_mape'][-1]:.2f}%\n\n")
        
        f.write("## Test Set Evaluation\n\n")
        f.write(f"- **Test Loss:** {metrics['loss']:.4f}\n")
        f.write(f"- **Test MAE:** {metrics['mae']:.4f}\n")
        f.write(f"- **Test MAPE:** {metrics['mape']:.2f}%\n\n")
        
        f.write("### Per-Horizon Metrics\n\n")
        f.write("| Horizon | RMSE | MAE | MAPE (%) |\n")
        f.write("|---------|------|-----|----------|\n")
        for horizon, hmetrics in metrics['horizon_metrics'].items():
            f.write(f"| {horizon} | {hmetrics['rmse']:.4f} | {hmetrics['mae']:.4f} | {hmetrics['mape']:.2f} |\n")
        f.write("\n")
        
        f.write("## Model Performance Analysis\n\n")
        
        # Analyze overfitting
        train_val_gap = history['val_loss'][-1] - history['loss'][-1]
        if train_val_gap < 0.1:
            f.write("- **Overfitting:** Low risk - training and validation losses are close\n")
        elif train_val_gap < 0.3:
            f.write("- **Overfitting:** Moderate - some gap between training and validation\n")
        else:
            f.write("- **Overfitting:** High risk - significant gap between training and validation\n")
        
        # Analyze convergence
        loss_improvement = history['val_loss'][0] - history['val_loss'][-1]
        improvement_pct = (loss_improvement / history['val_loss'][0]) * 100
        f.write(f"- **Convergence:** Validation loss improved by {improvement_pct:.1f}% from initial\n")
        
        # Early stopping analysis
        best_epoch = history['val_loss'].index(min(history['val_loss'])) + 1
        f.write(f"- **Best Epoch:** {best_epoch} (out of {len(history['loss'])} trained)\n\n")
        
        f.write("## Recommendations\n\n")
        
        if train_val_gap > 0.3:
            f.write("- Consider increasing dropout rate or adding regularization to reduce overfitting\n")
        
        if metrics['mape'] > 20:
            f.write("- MAPE is relatively high - consider feature engineering or data quality improvements\n")
        
        if len(history['loss']) < 20:
            f.write("- Training stopped early - model may benefit from more training data\n")
        
        f.write("\n## Files Generated\n\n")
        f.write(f"- Model: `lstm_model_{timestamp}.keras`\n")
        f.write(f"- Latest Model: `lstm_model_latest.keras`\n")
        f.write(f"- Training History: `training_history_{timestamp}.json`\n")
        f.write(f"- Evaluation Metrics: `evaluation_metrics_{timestamp}.json`\n")
        f.write(f"- Model Config: `model_config_{timestamp}.json`\n")
        f.write(f"- Training Curves: `training_curves_{timestamp}.png`\n")
        f.write(f"- This Report: `training_report_{timestamp}.md`\n")
    
    logger.info(f"Training report saved to {report_path}")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train LSTM Transfer Value Predictor')
    parser.add_argument(
        '--data',
        type=str,
        default='data/training/training_dataset.csv',
        help='Path to training dataset CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/lstm',
        help='Output directory for model and artifacts'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=10,
        help='Length of input sequences'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Maximum number of training epochs'
    )
    
    args = parser.parse_args()
    
    try:
        logger.info("=" * 80)
        logger.info("LSTM Transfer Value Predictor - Training Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Load data
        df = load_training_data(args.data)
        
        # Step 2: Prepare temporal data
        df = prepare_temporal_data(df)
        
        # Step 3: Select features
        feature_columns = select_features(df)
        
        # Step 4: Prepare sequences
        X, y, player_ids, splits = prepare_sequences_for_training(
            df=df,
            feature_columns=feature_columns,
            target_column='market_value',
            sequence_length=args.sequence_length,
            prediction_horizons=[1, 3, 6]
        )
        
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        X_test, y_test = splits['test']
        
        # Step 5: Train model
        model, history = train_lstm_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )
        
        # Step 6: Evaluate model
        metrics = evaluate_model(model, X_test, y_test)
        
        # Step 7: Save model and artifacts
        save_model_and_artifacts(
            model=model,
            history=history,
            metrics=metrics,
            output_dir=args.output
        )
        
        # Step 8: Generate visualizations
        generate_training_visualizations(
            history=history,
            output_dir=args.output
        )
        
        # Step 9: Generate report
        generate_training_report(
            metrics=metrics,
            history=history,
            output_dir=args.output
        )
        
        logger.info("=" * 80)
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Model and artifacts saved to: {args.output}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
