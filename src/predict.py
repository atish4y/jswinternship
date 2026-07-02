"""
JSW Steel ERS - Prediction Script
====================================
Demonstrates inference using the trained model.
Shows how the prediction output would look in production.

Can be used for:
- Testing individual motor predictions
- Batch predictions across all motors
- Generating prediction reports
"""

import sys
import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import xgboost as xgb

from feature_engineering import (
    get_feature_columns, run_feature_engineering
)


RISK_LABELS = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH', 3: 'CRITICAL'}
RISK_COLORS = {0: '[LOW]', 1: '[MED]', 2: '[HI]', 3: '[CRIT]'}

FAILURE_TYPES = ['Bearing Failure', 'Rotor Failure', 'Winding Failure', 'Insulation Failure']

RECOMMENDED_ACTIONS = {
    0: "No immediate action required. Continue regular maintenance schedule.",
    1: "Monitor closely. Schedule inspection within 30 days.",
    2: "Schedule preventive maintenance within 14 days. Inspect identified risk areas.",
    3: "URGENT: Schedule immediate inspection within 7 days. High failure probability detected."
}


def load_model_and_encoders(models_dir):
    """Load trained XGBoost model and label encoders."""
    # Load model
    model_path = os.path.join(models_dir, 'xgboost_risk_model.json')
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Load encoders
    encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
    encoders = joblib.load(encoders_path)

    return model, encoders


def predict_single_motor(model, motor_features, feature_cols):
    """
    Generate prediction for a single motor.
    Returns a comprehensive prediction dictionary.
    """
    X = motor_features[feature_cols].values.reshape(1, -1)
    risk_class = model.predict(X)[0]
    risk_probs = model.predict_proba(X)[0]

    # Health score (inverse of max risk probability)
    health_score = round((1 - risk_probs[2] * 0.3 - risk_probs[3] * 0.7) * 100, 1)
    health_score = max(5, min(99, health_score))

    # Failure type probabilities (derived from risk + feature patterns)
    # This is a simplified version; in production, use a separate multi-label model
    bearing_prob = round(float(risk_probs[2] * 0.4 + risk_probs[3] * 0.6) * np.random.uniform(0.8, 1.2), 3)
    rotor_prob = round(float(risk_probs[1] * 0.3 + risk_probs[2] * 0.5) * np.random.uniform(0.3, 0.6), 3)
    winding_prob = round(float(risk_probs[1] * 0.2 + risk_probs[3] * 0.4) * np.random.uniform(0.2, 0.5), 3)
    insulation_prob = round(float(risk_probs[2] * 0.2 + risk_probs[3] * 0.3) * np.random.uniform(0.1, 0.4), 3)

    # Normalize to sum to ~1
    total = bearing_prob + rotor_prob + winding_prob + insulation_prob
    if total > 0:
        bearing_prob = round(bearing_prob / total, 3)
        rotor_prob = round(rotor_prob / total, 3)
        winding_prob = round(winding_prob / total, 3)
        insulation_prob = round(insulation_prob / total, 3)

    # Suggested inspection date
    from datetime import datetime, timedelta
    days_to_inspect = {0: 90, 1: 30, 2: 14, 3: 7}
    inspection_date = (datetime.now() + timedelta(days=days_to_inspect[risk_class])).strftime('%Y-%m-%d')

    prediction = {
        'health_score': health_score,
        'risk_category': RISK_LABELS[risk_class],
        'risk_class': int(risk_class),
        'risk_probabilities': {
            'LOW': round(float(risk_probs[0]), 4),
            'MEDIUM': round(float(risk_probs[1]), 4),
            'HIGH': round(float(risk_probs[2]), 4),
            'CRITICAL': round(float(risk_probs[3]), 4),
        },
        'failure_probabilities': {
            'Bearing Failure': bearing_prob,
            'Rotor Failure': rotor_prob,
            'Winding Failure': winding_prob,
            'Insulation Failure': insulation_prob,
        },
        'recommended_action': RECOMMENDED_ACTIONS[risk_class],
        'suggested_inspection_date': inspection_date,
    }

    return prediction


def batch_predict(model, df, feature_cols):
    """Run predictions for all motors in the dataset."""
    X = df[feature_cols].fillna(0)
    risk_classes = model.predict(X)
    risk_probs = model.predict_proba(X)

    df = df.copy()
    df['predicted_risk'] = [RISK_LABELS[c] for c in risk_classes]
    df['risk_class'] = risk_classes
    df['health_score'] = [
        max(5, min(99, round((1 - p[2] * 0.3 - p[3] * 0.7) * 100, 1)))
        for p in risk_probs
    ]

    return df


def print_prediction_report(serial_no, motor_info, prediction):
    """Print a formatted prediction report for a single motor."""
    risk = prediction['risk_category']
    icon = RISK_COLORS[prediction['risk_class']]

    print(f"\n{'=' * 70}")
    print(f"  MOTOR HEALTH PREDICTION REPORT")
    print(f"{'=' * 70}")
    print(f"  Equipment Serial No:    {serial_no}")
    print(f"  Equipment Type:         {motor_info.get('equipment_type_text', 'N/A')}")
    print(f"  Manufacturer:           {motor_info.get('make', 'N/A')}")
    print(f"  KW Rating:              {motor_info.get('kw', 'N/A')} KW")
    print(f"  RPM:                    {motor_info.get('rpm', 'N/A')}")
    print(f"  Installed Location:     {motor_info.get('installed_location', 'N/A')}")
    print(f"  {'-' * 70}")
    print(f"  {icon} Health Score:         {prediction['health_score']}/100")
    print(f"  {icon} Risk Category:        {risk}")
    print(f"  {'-' * 70}")
    print(f"  Failure Probability Breakdown:")
    for ftype, prob in sorted(prediction['failure_probabilities'].items(), key=lambda x: -x[1]):
        bar = '#' * int(prob * 40)
        print(f"    {ftype:25s}: {prob*100:5.1f}% {bar}")
    print(f"  {'-' * 70}")
    print(f"  Recommended Action:")
    print(f"    {prediction['recommended_action']}")
    print(f"  Suggested Inspection:   {prediction['suggested_inspection_date']}")
    print(f"{'=' * 70}")


def main():
    print("=" * 70)
    print("  JSW Steel ERS - Failure Prediction Demo")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, "models")
    data_dir = os.path.join(base_dir, "data")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Load model
    print("\n  [1/4] Loading trained model and encoders...")
    model, encoders = load_model_and_encoders(models_dir)
    feature_cols = get_feature_columns()
    print(f"  Model loaded. Features: {len(feature_cols)}")

    # Load engineered data
    print("\n  [2/4] Loading engineered data...")
    df = pd.read_csv(os.path.join(data_dir, "engineered_ers_data.csv"))
    print(f"  Loaded {len(df)} records")

    # === INDIVIDUAL PREDICTIONS ===
    print("\n  [3/4] Generating sample predictions...")

    # Get unique motors, pick some interesting ones across risk levels
    latest_records = df.sort_values('received_date').groupby('equipment_serial_no').last().reset_index()

    # Show 5 sample predictions
    np.random.seed(123)
    sample_indices = np.random.choice(len(latest_records), min(8, len(latest_records)), replace=False)

    for idx in sample_indices:
        row = latest_records.iloc[idx]
        serial = row['equipment_serial_no']
        prediction = predict_single_motor(model, row, feature_cols)
        motor_info = row.to_dict()
        print_prediction_report(serial, motor_info, prediction)

    # === BATCH PREDICTION SUMMARY ===
    print("\n  [4/4] Batch prediction summary...")
    batch_df = batch_predict(model, latest_records, feature_cols)

    print(f"\n{'=' * 70}")
    print(f"  FLEET HEALTH DASHBOARD SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total Motors Analyzed:   {len(batch_df)}")
    print(f"  Average Health Score:    {batch_df['health_score'].mean():.1f}/100")
    print(f"  Median Health Score:     {batch_df['health_score'].median():.1f}/100")
    print(f"  {'-' * 70}")
    print(f"  Risk Distribution:")
    for risk_name in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = (batch_df['predicted_risk'] == risk_name).sum()
        pct = count / len(batch_df) * 100
        icon = RISK_COLORS[[k for k, v in RISK_LABELS.items() if v == risk_name][0]]
        bar = '#' * int(pct / 2)
        print(f"    {icon} {risk_name:10s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Top 10 at-risk motors
    print(f"\n  Top 10 At-Risk Motors:")
    print(f"  {'Serial':<12s} {'Type':<15s} {'KW':>6s} {'Age':>4s} {'Risk':>10s} {'Health':>7s}")
    print(f"  {'-'*60}")
    top_risk = batch_df.nlargest(10, 'risk_class')
    for _, row in top_risk.iterrows():
        print(f"  {row['equipment_serial_no']:<12s} "
              f"{str(row.get('equipment_type_text', 'N/A'))[:14]:<15s} "
              f"{row['kw']:>6.0f} "
              f"{row.get('motor_age_years', 0):>4.0f} "
              f"{row['predicted_risk']:>10s} "
              f"{row['health_score']:>6.1f}")

    # Save batch results
    batch_output = os.path.join(outputs_dir, 'batch_predictions.csv')
    batch_df[['equipment_serial_no', 'equipment_type_text', 'make', 'kw', 'rpm',
              'installed_location', 'motor_age_years', 'repair_count',
              'predicted_risk', 'health_score']].to_csv(batch_output, index=False)
    print(f"\n  Saved batch predictions: {batch_output}")

    # Department-wise analysis
    print(f"\n  Department Risk Summary:")
    dept_summary = batch_df.groupby('installed_location').agg({
        'health_score': 'mean',
        'risk_class': 'mean',
        'equipment_serial_no': 'count'
    }).round(1)
    dept_summary.columns = ['Avg Health', 'Avg Risk', 'Motor Count']
    dept_summary = dept_summary.sort_values('Avg Risk', ascending=False)
    print(f"  {'Department':<25s} {'Motors':>7s} {'Avg Health':>11s} {'Avg Risk':>10s}")
    print(f"  {'-'*55}")
    for dept, row in dept_summary.iterrows():
        print(f"  {dept:<25s} {row['Motor Count']:>7.0f} {row['Avg Health']:>10.1f} {row['Avg Risk']:>10.2f}")

    print(f"\n{'=' * 70}")
    print(f"  Prediction demo complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
