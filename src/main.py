"""
TransferIQ — Full pipeline runner
Runs all stages end-to-end.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("\n" + "="*60)
    print("  TRANSFERIQ — FULL PIPELINE")
    print("="*60)

    print("\n[Step 1/5] Data Cleaning...")
    from data_cleaning import run_cleaning
    run_cleaning()

    print("\n[Step 2/5] Feature Engineering...")
    from feature_engineering import run_feature_engineering
    run_feature_engineering()

    print("\n[Step 3/5] LSTM Model Training...")
    from lstm_model import run_lstm
    run_lstm()

    print("\n[Step 4/5] Ensemble Model (v1)...")
    from ensemble_model import run_ensemble
    run_ensemble()

    print("\n[Step 5/5] Best Model v2 (R²=0.76)...")
    from best_model import train
    train()

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE!")
    print("  Models → models/")
    print("  Results → outputs/")
    print("="*60)
    print("\nPredict a player:")
    print("  python src/best_model.py predict 'Bruno Fernandes'")
    print("  python src/predict.py --player 'Casemiro'")
    print("\nRun web app:")
    print("  cd webapp && python app.py")


if __name__ == '__main__':
    main()
