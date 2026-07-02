"""
JSW Steel ERS - Model Training Pipeline
=========================================
Trains, evaluates, and compares XGBoost vs Random Forest classifiers
for motor failure risk prediction.

Models:
1. XGBoost Classifier (primary) — multi-class risk prediction
2. Random Forest Classifier (baseline) — comparison model

Outputs:
- Trained model files (.json for XGBoost, .pkl for RF)
- Classification reports
- Confusion matrices
- ROC curves
- Feature importance plots
- SHAP analysis
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import xgboost as xgb
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import SHAP (optional but preferred)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("  WARNING: SHAP not installed. Skipping SHAP analysis.")

from feature_engineering import get_feature_columns


# ============================================================================
# CONFIGURATION
# ============================================================================

RISK_LABELS = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH', 3: 'CRITICAL'}
TARGET_COL = 'risk_category'
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_CV_FOLDS = 5

# XGBoost hyperparameters
XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'multi:softprob',
    'num_class': 4,
    'eval_metric': 'mlogloss',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0,
}

# Random Forest hyperparameters
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'class_weight': 'balanced',
}


def load_data():
    """Load engineered dataset."""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "engineered_ers_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Engineered data not found at {data_path}. "
            "Run feature_engineering.py first!"
        )
    return pd.read_csv(data_path)


def prepare_data(df):
    """Prepare feature matrix X and target y."""
    feature_cols = get_feature_columns()
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    # Handle any NaN
    X = X.fillna(X.median())

    return X, y, feature_cols


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier."""
    print("\n  Training XGBoost Classifier...")

    model = xgb.XGBClassifier(**XGB_PARAMS)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest classifier."""
    print("  Training Random Forest Classifier...")

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model and return metrics dict.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')

    # ROC AUC (one-vs-rest)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    try:
        roc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='weighted')
    except ValueError:
        roc = 0.0

    # Classification report
    report = classification_report(y_test, y_pred, target_names=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])

    metrics = {
        'model_name': model_name,
        'accuracy': round(acc, 4),
        'f1_macro': round(f1_macro, 4),
        'f1_weighted': round(f1_weighted, 4),
        'precision_weighted': round(prec, 4),
        'recall_weighted': round(rec, 4),
        'roc_auc_weighted': round(roc, 4),
        'classification_report': report,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_prob': y_prob,
    }

    return metrics


def cross_validate_model(model, X, y, model_name):
    """Run stratified k-fold cross-validation."""
    print(f"  Running {N_CV_FOLDS}-fold cross-validation for {model_name}...")
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted', n_jobs=-1)
    return {
        'cv_mean': round(scores.mean(), 4),
        'cv_std': round(scores.std(), 4),
        'cv_scores': [round(s, 4) for s in scores],
    }


def plot_confusion_matrix(cm, model_name, output_dir):
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
        yticklabels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(y_test, y_prob, model_name, output_dir):
    """Plot multi-class ROC curves."""
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    n_classes = 4
    class_names = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_name} - ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(model, feature_cols, model_name, output_dir):
    """Plot feature importance bar chart."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return

    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))
    bars = ax.barh(range(len(sorted_features)), sorted_importances, color=colors)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'{model_name} - Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, sorted_importances):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_shap_summary(model, X_test, feature_cols, output_dir):
    """Generate SHAP summary plot for XGBoost."""
    if not HAS_SHAP:
        return

    print("  Generating SHAP analysis (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    # For multi-class, shap_values is a list; use class index 2 (HIGH risk) for summary
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[2], X_test, feature_names=feature_cols,
                          show=False, max_display=18)
    else:
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                          show=False, max_display=18)
    plt.title('SHAP Feature Impact - HIGH Risk Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'shap_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_model_comparison(xgb_metrics, rf_metrics, output_dir):
    """Plot side-by-side model comparison."""
    metrics_to_compare = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'roc_auc_weighted']
    labels = ['Accuracy', 'F1 (Weighted)', 'Precision', 'Recall', 'ROC-AUC']

    xgb_vals = [xgb_metrics[m] for m in metrics_to_compare]
    rf_vals = [rf_metrics[m] for m in metrics_to_compare]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, xgb_vals, width, label='XGBoost', color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + width/2, rf_vals, width, label='Random Forest', color='#2ecc71', edgecolor='white')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison - XGBoost vs Random Forest', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def save_models(xgb_model, rf_model, feature_cols, xgb_metrics, rf_metrics, models_dir):
    """Save trained models and metadata."""
    # Save XGBoost model
    xgb_path = os.path.join(models_dir, 'xgboost_risk_model.json')
    xgb_model.save_model(xgb_path)
    print(f"  Saved XGBoost model: {xgb_path}")

    # Save Random Forest model
    rf_path = os.path.join(models_dir, 'random_forest_risk_model.pkl')
    joblib.dump(rf_model, rf_path)
    print(f"  Saved Random Forest model: {rf_path}")

    # Save model metadata
    metadata = {
        'primary_model': 'XGBoost',
        'feature_columns': feature_cols,
        'target': TARGET_COL,
        'risk_labels': RISK_LABELS,
        'n_features': len(feature_cols),
        'xgboost': {
            'params': {k: v for k, v in XGB_PARAMS.items() if k != 'eval_metric'},
            'metrics': {k: v for k, v in xgb_metrics.items()
                        if k not in ['classification_report', 'confusion_matrix', 'y_pred', 'y_prob']},
        },
        'random_forest': {
            'params': RF_PARAMS,
            'metrics': {k: v for k, v in rf_metrics.items()
                        if k not in ['classification_report', 'confusion_matrix', 'y_pred', 'y_prob']},
        },
    }
    meta_path = os.path.join(models_dir, 'model_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved metadata: {meta_path}")


def main():
    print("=" * 70)
    print("  JSW Steel ERS - Model Training Pipeline")
    print("=" * 70)

    # Setup directories
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # Load data
    print("\n  [1/8] Loading engineered data...")
    df = load_data()
    X, y, feature_cols = prepare_data(df)
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Target distribution:")
    for val, label in RISK_LABELS.items():
        count = (y == val).sum()
        print(f"    {label:10s}: {count:4d} ({count/len(y)*100:.1f}%)")

    # Split data
    print("\n  [2/8] Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # Train models
    print("\n  [3/8] Training models...")
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate models
    print("\n  [4/8] Evaluating models...")
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Print results
    print("\n" + "=" * 70)
    print("  RESULTS - XGBoost")
    print("=" * 70)
    print(f"  Accuracy:          {xgb_metrics['accuracy']:.4f}")
    print(f"  F1 (Weighted):     {xgb_metrics['f1_weighted']:.4f}")
    print(f"  F1 (Macro):        {xgb_metrics['f1_macro']:.4f}")
    print(f"  Precision:         {xgb_metrics['precision_weighted']:.4f}")
    print(f"  Recall:            {xgb_metrics['recall_weighted']:.4f}")
    print(f"  ROC-AUC:           {xgb_metrics['roc_auc_weighted']:.4f}")
    print(f"\n{xgb_metrics['classification_report']}")

    print("=" * 70)
    print("  RESULTS - Random Forest")
    print("=" * 70)
    print(f"  Accuracy:          {rf_metrics['accuracy']:.4f}")
    print(f"  F1 (Weighted):     {rf_metrics['f1_weighted']:.4f}")
    print(f"  F1 (Macro):        {rf_metrics['f1_macro']:.4f}")
    print(f"  Precision:         {rf_metrics['precision_weighted']:.4f}")
    print(f"  Recall:            {rf_metrics['recall_weighted']:.4f}")
    print(f"  ROC-AUC:           {rf_metrics['roc_auc_weighted']:.4f}")
    print(f"\n{rf_metrics['classification_report']}")

    # Cross-validation
    print("\n  [5/8] Cross-validation...")
    xgb_cv = cross_validate_model(
        xgb.XGBClassifier(**XGB_PARAMS), X, y, "XGBoost"
    )
    rf_cv = cross_validate_model(
        RandomForestClassifier(**RF_PARAMS), X, y, "Random Forest"
    )
    print(f"  XGBoost CV F1:       {xgb_cv['cv_mean']:.4f} ± {xgb_cv['cv_std']:.4f}")
    print(f"  Random Forest CV F1: {rf_cv['cv_mean']:.4f} ± {rf_cv['cv_std']:.4f}")

    # Generate plots
    print("\n  [6/8] Generating visualizations...")
    plot_confusion_matrix(xgb_metrics['confusion_matrix'], "XGBoost", outputs_dir)
    plot_confusion_matrix(rf_metrics['confusion_matrix'], "Random Forest", outputs_dir)
    plot_roc_curves(y_test, xgb_metrics['y_prob'], "XGBoost", outputs_dir)
    plot_roc_curves(y_test, rf_metrics['y_prob'], "Random Forest", outputs_dir)
    plot_feature_importance(xgb_model, feature_cols, "XGBoost", outputs_dir)
    plot_feature_importance(rf_model, feature_cols, "Random Forest", outputs_dir)
    plot_model_comparison(xgb_metrics, rf_metrics, outputs_dir)

    # SHAP analysis (XGBoost only)
    print("\n  [7/8] SHAP analysis...")
    plot_shap_summary(xgb_model, X_test, feature_cols, outputs_dir)

    # Save models
    print("\n  [8/8] Saving models and artifacts...")
    save_models(xgb_model, rf_model, feature_cols, xgb_metrics, rf_metrics, models_dir)

    # Save classification report to text
    report_path = os.path.join(outputs_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("JSW Steel ERS - Model Training Results\n")
        f.write("=" * 70 + "\n\n")
        f.write("XGBoost Results:\n")
        f.write(f"  Accuracy: {xgb_metrics['accuracy']:.4f}\n")
        f.write(f"  F1 (Weighted): {xgb_metrics['f1_weighted']:.4f}\n")
        f.write(f"  ROC-AUC: {xgb_metrics['roc_auc_weighted']:.4f}\n")
        f.write(f"  CV F1: {xgb_cv['cv_mean']:.4f} ± {xgb_cv['cv_std']:.4f}\n\n")
        f.write(xgb_metrics['classification_report'])
        f.write("\n\nRandom Forest Results:\n")
        f.write(f"  Accuracy: {rf_metrics['accuracy']:.4f}\n")
        f.write(f"  F1 (Weighted): {rf_metrics['f1_weighted']:.4f}\n")
        f.write(f"  ROC-AUC: {rf_metrics['roc_auc_weighted']:.4f}\n")
        f.write(f"  CV F1: {rf_cv['cv_mean']:.4f} ± {rf_cv['cv_std']:.4f}\n\n")
        f.write(rf_metrics['classification_report'])
    print(f"  Saved report: {report_path}")

    # Final summary
    winner = "XGBoost" if xgb_metrics['f1_weighted'] >= rf_metrics['f1_weighted'] else "Random Forest"
    print(f"\n{'=' * 70}")
    print(f"  RECOMMENDED MODEL: {winner}")
    print(f"  XGBoost F1={xgb_metrics['f1_weighted']:.4f} vs RF F1={rf_metrics['f1_weighted']:.4f}")
    print(f"{'=' * 70}")

    return xgb_model, rf_model


if __name__ == "__main__":
    main()
