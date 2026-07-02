"""
JSW Steel ERS - Main Pipeline Runner
======================================
Runs the complete ML pipeline end-to-end:
1. Generate dummy data
2. Feature engineering
3. Model training & evaluation
4. ONNX export
5. Prediction demo
"""

import os
import sys
import time

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))


def main():
    start_time = time.time()

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   JSW STEEL - ERS PREDICTIVE MAINTENANCE ML PIPELINE            #")
    print("#   Electrical Repair Shop - AI/ML Integration                     #")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    # Step 1: Generate Dummy Data
    print("\n\n" + "=" * 70)
    print("  STEP 1/5: GENERATING SYNTHETIC ERS DATA")
    print("=" * 70)
    from generate_dummy_data import main as generate_data
    generate_data()

    # Step 2: Feature Engineering
    print("\n\n" + "=" * 70)
    print("  STEP 2/5: FEATURE ENGINEERING")
    print("=" * 70)
    from feature_engineering import main as engineer_features
    engineer_features()

    # Step 3: Model Training
    print("\n\n" + "=" * 70)
    print("  STEP 3/5: MODEL TRAINING & EVALUATION")
    print("=" * 70)
    from train_model import main as train_models
    train_models()

    # Step 4: ONNX Export
    print("\n\n" + "=" * 70)
    print("  STEP 4/5: ONNX MODEL EXPORT")
    print("=" * 70)
    try:
        from export_onnx import main as export_onnx
        export_onnx()
    except Exception as e:
        print(f"  WARNING: ONNX export failed: {e}")
        print("  This is non-critical. The .json model can still be used.")
        print("  Install onnxmltools for ONNX support: pip install onnxmltools onnxruntime")

    # Step 5: Prediction Demo
    print("\n\n" + "=" * 70)
    print("  STEP 5/5: PREDICTION DEMO")
    print("=" * 70)
    from predict import main as run_predictions
    run_predictions()

    # Final Summary
    elapsed = time.time() - start_time
    print("\n\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   PIPELINE COMPLETE                                              #")
    print(f"#   Total Time: {elapsed:.1f}s" + " " * (52 - len(f"{elapsed:.1f}")) + "#")
    print("#" + " " * 68 + "#")
    print("#   Outputs generated:                                             #")
    print("#     -- data/dummy_ers_data.csv         (synthetic dataset)       #")
    print("#     -- data/engineered_ers_data.csv    (feature-engineered)      #")
    print("#     -- models/xgboost_risk_model.json  (trained XGBoost)         #")
    print("#     -- models/random_forest_model.pkl  (trained RF)              #")
    print("#     -- outputs/confusion_matrix_*.png  (confusion matrices)      #")
    print("#     -- outputs/roc_curve_*.png         (ROC curves)              #")
    print("#     -- outputs/feature_importance_*.png(feature importance)      #")
    print("#     -- outputs/model_comparison.png    (XGB vs RF comparison)    #")
    print("#     -- outputs/shap_summary.png        (SHAP analysis)           #")
    print("#     -- outputs/classification_report.txt                         #")
    print("#     -- outputs/batch_predictions.csv   (fleet predictions)       #")
    print("#" + " " * 68 + "#")
    print("#" * 70)


if __name__ == "__main__":
    main()
