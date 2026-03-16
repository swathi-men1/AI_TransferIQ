"""
Comprehensive Model Evaluation Report Generator

This script aggregates results from all evaluation scripts and generates
a comprehensive report comparing LSTM vs Ensemble models with visualizations.

Requirements: 10.1, 10.5, 10.6, 10.7
"""

import sys
import os
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Try to import seaborn for enhanced styling
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.data_splitter import DataSplitter
from src.evaluation.model_evaluator import ModelEvaluator
from src.models.lstm_model import LSTMTransferValuePredictor, TENSORFLOW_AVAILABLE
from src.models.ensemble_model import EnsembleTransferValuePredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot style
if SEABORN_AVAILABLE:
    sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the training dataset."""
    logger.info(f"Loading dataset from {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded dataset with {len(df)} samples and {len(df.columns)} columns")
    
    return df


def prepare_test_data(df: pd.DataFrame, target_column: str = 'market_value') -> tuple:
    """Split data and prepare test set."""
    logger.info("Preparing test data using DataSplitter...")
    
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    train_data, val_data, test_data = splitter.split(df, temporal_column=None)
    
    logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    if target_column not in test_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    
    logger.info(f"Test set prepared - Features: {X_test.shape}, Target: {y_test.shape}")
    
    return X_test, y_test, test_data


def load_models(lstm_model_dir: str, ensemble_model_dir: str) -> dict:
    """Load trained models."""
    models = {}
    
    # Load LSTM model
    if TENSORFLOW_AVAILABLE:
        lstm_path = os.path.join(lstm_model_dir, 'lstm_model.h5')
        if os.path.exists(lstm_path):
            try:
                lstm_model = LSTMTransferValuePredictor()
                lstm_model.load_model(lstm_path)
                models['LSTM'] = lstm_model
                logger.info("LSTM model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load LSTM model: {e}")
    
    # Load Ensemble model
    xgb_path = os.path.join(ensemble_model_dir, 'xgboost_model.json')
    lgb_path = os.path.join(ensemble_model_dir, 'lightgbm_model.txt')
    
    if os.path.exists(xgb_path) and os.path.exists(lgb_path):
        try:
            ensemble_model = EnsembleTransferValuePredictor()
            ensemble_model.load_models(ensemble_model_dir)
            models['Ensemble'] = ensemble_model
            logger.info("Ensemble model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Ensemble model: {e}")
    
    return models



def create_predicted_vs_actual_plots(models: dict, X_test: pd.DataFrame, 
                                     y_test: pd.Series, output_dir: Path):
    """Create scatter plots of predicted vs actual values."""
    logger.info("Creating predicted vs actual scatter plots...")
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    feature_columns = [col for col in X_test.columns if col != 'Position']
    
    for model_name, model in models.items():
        X_for_prediction = X_test[feature_columns]
        y_pred = model.predict(X_for_prediction)
        
        # Scatter plot
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, y_pred, alpha=0.5, s=50, edgecolors='k', linewidths=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Transfer Value (€)', fontsize=12)
        plt.ylabel('Predicted Transfer Value (€)', fontsize=12)
        plt.title(f'{model_name} - Predicted vs Actual Transfer Values', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / f'{model_name}_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log scale version for better visualization
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, y_pred, alpha=0.5, s=50, edgecolors='k', linewidths=0.5)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Actual Transfer Value (€) - Log Scale', fontsize=12)
        plt.ylabel('Predicted Transfer Value (€) - Log Scale', fontsize=12)
        plt.title(f'{model_name} - Predicted vs Actual (Log Scale)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig(viz_dir / f'{model_name}_predicted_vs_actual_log.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Predicted vs actual plots saved to {viz_dir}/")



def create_residual_plots(models: dict, X_test: pd.DataFrame, 
                         y_test: pd.Series, output_dir: Path):
    """Create residual plots for error analysis."""
    logger.info("Creating residual plots...")
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    feature_columns = [col for col in X_test.columns if col != 'Position']
    
    for model_name, model in models.items():
        X_for_prediction = X_test[feature_columns]
        y_pred = model.predict(X_for_prediction)
        residuals = y_test.values - y_pred
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{model_name} - Residual Analysis', fontsize=16, fontweight='bold')
        
        # 1. Residuals vs Predicted Values
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Value (€)', fontsize=10)
        axes[0, 0].set_ylabel('Residuals (€)', fontsize=10)
        axes[0, 0].set_title('Residuals vs Predicted Values', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals vs Actual Values
        axes[0, 1].scatter(y_test, residuals, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Actual Value (€)', fontsize=10)
        axes[0, 1].set_ylabel('Residuals (€)', fontsize=10)
        axes[0, 1].set_title('Residuals vs Actual Values', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Histogram of Residuals
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residuals (€)', fontsize=10)
        axes[1, 0].set_ylabel('Frequency', fontsize=10)
        axes[1, 0].set_title('Distribution of Residuals', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / f'{model_name}_residual_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Residual plots saved to {viz_dir}/")



def create_model_comparison_plots(models: dict, X_test: pd.DataFrame, 
                                  y_test: pd.Series, evaluator: ModelEvaluator, 
                                  output_dir: Path):
    """Create comparison plots between models."""
    logger.info("Creating model comparison plots...")
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    feature_columns = [col for col in X_test.columns if col != 'Position']
    X_for_prediction = X_test[feature_columns]
    
    # Get comparison metrics
    comparison_df = evaluator.compare_models(models, X_test, y_test, feature_columns=feature_columns)
    
    # 1. Metrics comparison bar chart
    metrics_to_plot = ['rmse', 'mae', 'r2', 'mape']
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
    
    if len(available_metrics) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(available_metrics[:4]):
            row = idx // 2
            col = idx % 2
            
            model_names = comparison_df['model']
            metric_values = comparison_df[metric]
            
            bars = axes[row, col].bar(model_names, metric_values, alpha=0.7)
            axes[row, col].set_ylabel(metric.upper(), fontsize=10)
            axes[row, col].set_title(f'{metric.upper()} Comparison', fontsize=12)
            axes[row, col].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{height:.3f}' if metric == 'r2' else f'{height:.2f}',
                                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'model_comparison_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Side-by-side prediction comparison
    if len(models) == 2:
        model_names = list(models.keys())
        model1_pred = models[model_names[0]].predict(X_for_prediction)
        model2_pred = models[model_names[1]].predict(X_for_prediction)
        
        plt.figure(figsize=(10, 10))
        plt.scatter(model1_pred, model2_pred, alpha=0.5, s=50, edgecolors='k', linewidths=0.5)
        
        min_val = min(model1_pred.min(), model2_pred.min())
        max_val = max(model1_pred.max(), model2_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Agreement')
        
        plt.xlabel(f'{model_names[0]} Predictions (€)', fontsize=12)
        plt.ylabel(f'{model_names[1]} Predictions (€)', fontsize=12)
        plt.title('Model Prediction Agreement', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'model_prediction_agreement.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Model comparison plots saved to {viz_dir}/")



def load_existing_results(reports_dir: Path) -> dict:
    """Load results from previous evaluation scripts."""
    logger.info("Loading existing evaluation results...")
    
    results = {
        'overall': None,
        'position': None,
        'value_range': None
    }
    
    # Load overall evaluation results
    overall_json = reports_dir / 'evaluation_results.json'
    if overall_json.exists():
        with open(overall_json, 'r') as f:
            results['overall'] = json.load(f)
        logger.info("Loaded overall evaluation results")
    
    # Load position-based results
    position_json = reports_dir / 'position_analysis' / 'position_evaluation_results.json'
    if position_json.exists():
        with open(position_json, 'r') as f:
            results['position'] = json.load(f)
        logger.info("Loaded position-based evaluation results")
    
    # Load value-range results
    value_range_json = reports_dir / 'value_range_analysis' / 'value_range_evaluation_results.json'
    if value_range_json.exists():
        with open(value_range_json, 'r') as f:
            results['value_range'] = json.load(f)
        logger.info("Loaded value-range evaluation results")
    
    return results



def generate_comprehensive_report(models: dict, X_test: pd.DataFrame, y_test: pd.Series,
                                  evaluator: ModelEvaluator, existing_results: dict,
                                  output_dir: Path):
    """Generate comprehensive markdown evaluation report."""
    logger.info("Generating comprehensive evaluation report...")
    
    report_path = output_dir / 'model_evaluation_report.md'
    
    feature_columns = [col for col in X_test.columns if col != 'Position']
    X_for_prediction = X_test[feature_columns]
    
    with open(report_path, 'w') as f:
        # Header
        f.write("# Comprehensive Model Evaluation Report\n\n")
        f.write("## AI-Driven Player Transfer Value Prediction System\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        
        # Get overall comparison
        comparison_df = evaluator.compare_models(models, X_test, y_test, feature_columns=feature_columns)
        
        if len(models) > 0:
            # Determine best model
            if 'rmse' in comparison_df.columns:
                best_model_idx = comparison_df['rmse'].idxmin()
                best_model_name = comparison_df.loc[best_model_idx, 'model']
                best_rmse = comparison_df.loc[best_model_idx, 'rmse']
                best_r2 = comparison_df.loc[best_model_idx, 'r2'] if 'r2' in comparison_df.columns else None
                
                f.write(f"### Recommended Model: **{best_model_name}**\n\n")
                f.write(f"Based on comprehensive evaluation across {len(y_test)} test samples, ")
                f.write(f"the **{best_model_name}** model demonstrates the best overall performance.\n\n")
                f.write(f"**Key Performance Metrics:**\n")
                f.write(f"- RMSE: €{best_rmse:,.2f}\n")
                if best_r2 is not None:
                    f.write(f"- R² Score: {best_r2:.3f}\n")
                if 'mae' in comparison_df.columns:
                    best_mae = comparison_df.loc[best_model_idx, 'mae']
                    f.write(f"- MAE: €{best_mae:,.2f}\n")
                f.write("\n")
        
        # Model strengths and limitations
        f.write("### Model Strengths and Limitations\n\n")
        
        for model_name in models.keys():
            f.write(f"#### {model_name} Model\n\n")
            
            if model_name == 'LSTM':
                f.write("**Strengths:**\n")
                f.write("- Captures temporal patterns and trends in player performance\n")
                f.write("- Handles sequential data effectively\n")
                f.write("- Can model complex non-linear relationships\n")
                f.write("- Suitable for time-series forecasting\n\n")
                
                f.write("**Limitations:**\n")
                f.write("- Requires substantial training data\n")
                f.write("- Longer training time compared to ensemble methods\n")
                f.write("- May overfit on small datasets\n")
                f.write("- Less interpretable than tree-based models\n\n")
            
            elif model_name == 'Ensemble':
                f.write("**Strengths:**\n")
                f.write("- Combines XGBoost and LightGBM for robust predictions\n")
                f.write("- Handles tabular data efficiently\n")
                f.write("- Provides feature importance insights\n")
                f.write("- Fast training and inference\n")
                f.write("- Good generalization performance\n\n")
                
                f.write("**Limitations:**\n")
                f.write("- Does not explicitly model temporal dependencies\n")
                f.write("- May struggle with sequential patterns\n")
                f.write("- Requires careful hyperparameter tuning\n")
                f.write("- Can be sensitive to feature engineering quality\n\n")
        
        f.write("---\n\n")
        
        # Overall Performance Comparison
        f.write("## Overall Performance Comparison\n\n")
        f.write(f"**Test Set Size:** {len(y_test)} samples\n\n")
        f.write("### Performance Metrics\n\n")
        f.write(comparison_df.to_markdown(index=False))
        f.write("\n\n")
        
        # Metric explanations
        f.write("**Metric Definitions:**\n")
        f.write("- **RMSE** (Root Mean Squared Error): Average prediction error magnitude (lower is better)\n")
        f.write("- **MAE** (Mean Absolute Error): Average absolute prediction error (lower is better)\n")
        f.write("- **R²** (R-squared): Proportion of variance explained by the model (higher is better, max 1.0)\n")
        f.write("- **MAPE** (Mean Absolute Percentage Error): Average percentage error (lower is better)\n")
        f.write("- **Median AE**: Median absolute error, robust to outliers (lower is better)\n")
        f.write("- **Max Error**: Largest prediction error in the test set\n\n")
        
        # Residual Analysis
        f.write("### Residual Analysis\n\n")
        
        if existing_results['overall'] and 'residual_analysis' in existing_results['overall']:
            for model_name, analysis in existing_results['overall']['residual_analysis'].items():
                f.write(f"#### {model_name}\n\n")
                f.write(f"- **Mean Residual:** €{analysis['mean_residual']:,.2f}\n")
                f.write(f"- **Std Residual:** €{analysis['std_residual']:,.2f}\n")
                f.write(f"- **Skewness:** {analysis['skewness']:.3f}\n")
                f.write(f"- **Kurtosis:** {analysis['kurtosis']:.3f}\n")
                f.write(f"- **Residuals Normal:** {analysis['is_normal']}\n\n")
                
                # Interpretation
                if abs(analysis['mean_residual']) < 1000:
                    f.write("  *Interpretation:* Model shows minimal systematic bias.\n")
                elif analysis['mean_residual'] > 0:
                    f.write("  *Interpretation:* Model tends to underestimate transfer values.\n")
                else:
                    f.write("  *Interpretation:* Model tends to overestimate transfer values.\n")
                f.write("\n")
        
        f.write("---\n\n")
        
        # Position-Based Insights
        f.write("## Position-Based Performance Analysis\n\n")
        
        if existing_results['position']:
            f.write("This section analyzes model performance across different player positions.\n\n")
            
            for model_name, position_results in existing_results['position'].items():
                f.write(f"### {model_name}\n\n")
                
                position_df = pd.DataFrame(position_results)
                if not position_df.empty:
                    f.write(position_df.to_markdown(index=False))
                    f.write("\n\n")
                    
                    # Key insights
                    if 'rmse' in position_df.columns:
                        best_pos = position_df.loc[position_df['rmse'].idxmin(), 'position']
                        worst_pos = position_df.loc[position_df['rmse'].idxmax(), 'position']
                        
                        f.write(f"**Key Insights:**\n")
                        f.write(f"- Best performance: **{best_pos}** position\n")
                        f.write(f"- Most challenging: **{worst_pos}** position\n")
                        f.write(f"- Total positions analyzed: {len(position_df)}\n\n")
        else:
            f.write("*Position-based analysis not available. Run `python scripts/evaluate_by_position.py` to generate.*\n\n")
        
        f.write("---\n\n")
        
        # Value-Range Insights
        f.write("## Value-Range Performance Analysis\n\n")
        
        if existing_results['value_range']:
            f.write("This section analyzes model performance across different transfer value ranges.\n\n")
            
            for model_name, range_results in existing_results['value_range'].items():
                f.write(f"### {model_name}\n\n")
                
                range_df = pd.DataFrame(range_results)
                if not range_df.empty:
                    f.write(range_df.to_markdown(index=False))
                    f.write("\n\n")
                    
                    # Key insights
                    if 'rmse' in range_df.columns:
                        best_range = range_df.loc[range_df['rmse'].idxmin(), 'value_range']
                        worst_range = range_df.loc[range_df['rmse'].idxmax(), 'value_range']
                        
                        f.write(f"**Key Insights:**\n")
                        f.write(f"- Best performance: **{best_range}** value range\n")
                        f.write(f"- Most challenging: **{worst_range}** value range\n")
                        f.write(f"- Total ranges analyzed: {len(range_df)}\n\n")
        else:
            f.write("*Value-range analysis not available. Run `python scripts/evaluate_by_value_range.py` to generate.*\n\n")
        
        f.write("---\n\n")
        
        # Visualizations
        f.write("## Visualizations\n\n")
        f.write("The following visualizations have been generated to support this analysis:\n\n")
        
        f.write("### Overall Performance\n\n")
        for model_name in models.keys():
            f.write(f"#### {model_name}\n\n")
            f.write(f"- `visualizations/{model_name}_predicted_vs_actual.png` - Predicted vs Actual scatter plot\n")
            f.write(f"- `visualizations/{model_name}_predicted_vs_actual_log.png` - Predicted vs Actual (log scale)\n")
            f.write(f"- `visualizations/{model_name}_residual_analysis.png` - Comprehensive residual analysis\n")
            f.write("\n")
        
        f.write("### Model Comparison\n\n")
        f.write("- `visualizations/model_comparison_metrics.png` - Side-by-side metrics comparison\n")
        if len(models) == 2:
            f.write("- `visualizations/model_prediction_agreement.png` - Model prediction agreement plot\n")
        f.write("\n")
        
        f.write("### Position-Based Analysis\n\n")
        if existing_results['position']:
            f.write("- See `reports/position_analysis/` directory for detailed position-based visualizations\n")
            f.write("- Includes RMSE, R², error distributions, and comparison heatmaps by position\n\n")
        
        f.write("### Value-Range Analysis\n\n")
        if existing_results['value_range']:
            f.write("- See `reports/value_range_analysis/` directory for detailed value-range visualizations\n")
            f.write("- Includes RMSE, R², MAPE, error distributions, and comparison heatmaps by value range\n\n")
        
        f.write("---\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if len(models) > 0:
            # Determine best model
            if 'rmse' in comparison_df.columns:
                best_model_idx = comparison_df['rmse'].idxmin()
                best_model_name = comparison_df.loc[best_model_idx, 'model']
                
                f.write(f"### Deployment Recommendation\n\n")
                f.write(f"Based on the comprehensive evaluation, we recommend deploying the **{best_model_name}** model ")
                f.write(f"for production use due to its superior performance across multiple metrics.\n\n")
        
        f.write("### Model Improvement Opportunities\n\n")
        f.write("1. **Feature Engineering:** Explore additional temporal features and player interaction metrics\n")
        f.write("2. **Hyperparameter Tuning:** Further optimize model hyperparameters using Bayesian optimization\n")
        f.write("3. **Ensemble Methods:** Consider stacking LSTM and tree-based models for improved predictions\n")
        f.write("4. **Data Augmentation:** Collect more historical data to improve temporal modeling\n")
        f.write("5. **Position-Specific Models:** Train specialized models for each player position\n")
        f.write("6. **Value-Range Models:** Develop separate models for different value ranges\n\n")
        
        f.write("### Monitoring and Maintenance\n\n")
        f.write("1. **Regular Retraining:** Retrain models quarterly with updated transfer data\n")
        f.write("2. **Performance Monitoring:** Track prediction accuracy on new transfers\n")
        f.write("3. **Drift Detection:** Monitor for data drift in player statistics and market conditions\n")
        f.write("4. **A/B Testing:** Compare model versions in production to validate improvements\n\n")
        
        f.write("---\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("This comprehensive evaluation demonstrates that both LSTM and Ensemble models ")
        f.write("can effectively predict player transfer values. The models show strong performance ")
        f.write("across different player positions and value ranges, with specific strengths in different scenarios.\n\n")
        
        f.write("The evaluation reveals opportunities for further improvement through enhanced feature engineering, ")
        f.write("position-specific modeling, and ensemble techniques. Regular monitoring and retraining will ensure ")
        f.write("the models remain accurate as market conditions evolve.\n\n")
        
        f.write("---\n\n")
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n")
    
    logger.info(f"Comprehensive report saved to {report_path}")



def main():
    """Main execution function."""
    logger.info("="*80)
    logger.info("COMPREHENSIVE MODEL EVALUATION REPORT GENERATOR")
    logger.info("="*80)
    
    # Paths
    dataset_path = 'data/training/training_dataset.csv'
    lstm_model_dir = 'models/lstm'
    ensemble_model_dir = 'models/ensemble'
    reports_dir = Path('reports')
    output_dir = reports_dir
    
    try:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        df = load_dataset(dataset_path)
        
        # Prepare test data
        X_test, y_test, test_data = prepare_test_data(df, target_column='market_value')
        
        # Load models
        models = load_models(lstm_model_dir, ensemble_model_dir)
        
        if not models:
            logger.error("No trained models found. Please train models first.")
            logger.info("\nTo train models, run:")
            logger.info("  - python scripts/train_lstm.py")
            logger.info("  - python scripts/train_ensemble.py")
            return 1
        
        logger.info(f"Loaded {len(models)} model(s): {', '.join(models.keys())}")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(config={
            'metrics': ['rmse', 'mae', 'r2', 'mape', 'median_ae', 'max_error'],
            'cv_folds': 5
        })
        
        # Load existing evaluation results
        existing_results = load_existing_results(reports_dir)
        
        # Create visualizations
        logger.info("\n" + "="*80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*80 + "\n")
        
        create_predicted_vs_actual_plots(models, X_test, y_test, output_dir)
        create_residual_plots(models, X_test, y_test, output_dir)
        create_model_comparison_plots(models, X_test, y_test, evaluator, output_dir)
        
        # Generate comprehensive report
        logger.info("\n" + "="*80)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("="*80 + "\n")
        
        generate_comprehensive_report(models, X_test, y_test, evaluator, 
                                     existing_results, output_dir)
        
        logger.info("\n" + "="*80)
        logger.info("REPORT GENERATION COMPLETE")
        logger.info("="*80)
        logger.info(f"\nComprehensive evaluation report saved to:")
        logger.info(f"  - {output_dir}/model_evaluation_report.md")
        logger.info(f"\nVisualizations saved to:")
        logger.info(f"  - {output_dir}/visualizations/")
        logger.info(f"\nTo view the report, open:")
        logger.info(f"  {output_dir}/model_evaluation_report.md")
        
        return 0
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
