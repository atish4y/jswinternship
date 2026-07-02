"""
JSW Steel ERS - Feature Engineering
=====================================
Transforms raw ERS requisition data into ML-ready features for
predictive maintenance models.

Key Feature Groups:
1. Motor Specifications (KW, RPM, Voltage, Current)
2. Motor Age & History (age, repair count, days since last repair)
3. Categorical Encodings (make, equipment type, duty cycle, application)
4. Aggregated Risk Signals (department failure rate, make failure rate)
5. Target Variable (risk_category derived from failure patterns)
"""

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def compute_motor_age(df):
    """Compute motor age in years from manufacturing year."""
    df = df.copy()
    df['motor_age_years'] = 2026 - df['motor_manufacturing_year']
    return df


def compute_repair_history(df):
    """
    Compute per-motor repair history features:
    - repair_count: total repairs for this motor up to this record
    - days_since_last_repair: days between this repair and previous one
    - previous_refurb_count: number of previous refurbishment jobs
    """
    df = df.copy()
    df['received_date'] = pd.to_datetime(df['received_date'])

    # Sort by motor and date for correct history computation
    df = df.sort_values(['equipment_serial_no', 'received_date']).reset_index(drop=True)

    # Repair count (cumulative per motor)
    df['repair_count'] = df.groupby('equipment_serial_no').cumcount() + 1

    # Days since last repair
    df['prev_repair_date'] = df.groupby('equipment_serial_no')['received_date'].shift(1)
    df['days_since_last_repair'] = (df['received_date'] - df['prev_repair_date']).dt.days
    df['days_since_last_repair'] = df['days_since_last_repair'].fillna(365 * 3)  # default: 3 years for first repair

    # Previous refurbishment count
    df['has_prev_refurb'] = (df['previous_refurbishment_job_no'] != '').astype(int)
    df['previous_refurb_count'] = df.groupby('equipment_serial_no')['has_prev_refurb'].cumsum()

    # Drop helper columns
    df.drop(columns=['prev_repair_date', 'has_prev_refurb'], inplace=True)

    return df


def compute_failure_history(df):
    """
    Compute per-motor failure type counts (cumulative up to current record).
    """
    df = df.copy()

    # Create binary failure type columns
    failure_types = ['Bearing Failure', 'Winding Failure', 'Rotor Failure', 'Insulation Failure']
    for ft in failure_types:
        col_name = ft.lower().replace(' ', '_') + '_count'
        df[f'_is_{ft}'] = (df['reason_of_failure_text'] == ft).astype(int)
        df[col_name] = df.groupby('equipment_serial_no')[f'_is_{ft}'].cumsum()
        df.drop(columns=[f'_is_{ft}'], inplace=True)

    return df


def compute_department_failure_rate(df):
    """
    Compute failure rate by department (installed_location).
    Failure rate = total failures / total motors in department.
    """
    df = df.copy()
    dept_total = df.groupby('installed_location')['requisition_id'].count()
    dept_motors = df.groupby('installed_location')['equipment_serial_no'].nunique()
    dept_rate = (dept_total / dept_motors).to_dict()
    df['dept_failure_rate'] = df['installed_location'].map(dept_rate)
    return df


def compute_make_failure_rate(df):
    """
    Compute failure rate by manufacturer.
    """
    df = df.copy()
    make_total = df.groupby('make')['requisition_id'].count()
    make_motors = df.groupby('make')['equipment_serial_no'].nunique()
    make_rate = (make_total / make_motors).to_dict()
    df['make_failure_rate'] = df['make'].map(make_rate)
    return df


def encode_categoricals(df, encoders=None, fit=True):
    """
    Label-encode categorical features.
    Returns df with encoded columns and the fitted encoders.
    """
    df = df.copy()
    categorical_cols = {
        'duty_cycle': 'duty_cycle_encoded',
        'make': 'make_encoded',
        'equipment_type_text': 'equipment_type_encoded',
        'application_use': 'application_use_encoded',
    }

    if encoders is None:
        encoders = {}

    for col, new_col in categorical_cols.items():
        if fit:
            le = LabelEncoder()
            df[new_col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels
            df[new_col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    return df, encoders


def create_risk_target(df):
    """
    Create the target variable: risk_category
    Based on a composite risk score derived from:
    - Motor age
    - Repair frequency
    - Days since last repair (inverse)
    - Failure history
    - KW rating

    Risk Categories: LOW (0), MEDIUM (1), HIGH (2), CRITICAL (3)
    """
    df = df.copy()

    # Normalize components to 0-1 range
    age_score = np.clip(df['motor_age_years'] / 30, 0, 1)  # 30 years = max age
    repair_freq_score = np.clip(df['repair_count'] / 8, 0, 1)  # 8 repairs = high
    recency_score = np.clip(1 - (df['days_since_last_repair'] / 1095), 0, 1)  # Recent repair = higher risk
    kw_score = np.clip(df['kw'] / 500, 0, 1)  # 500 KW = max
    failure_history_score = np.clip(
        (df['bearing_failure_count'] + df['winding_failure_count'] +
         df['rotor_failure_count'] + df['insulation_failure_count']) / 6, 0, 1
    )

    # Weighted composite score
    risk_score = (
        0.25 * age_score +
        0.25 * repair_freq_score +
        0.15 * recency_score +
        0.15 * kw_score +
        0.20 * failure_history_score
    )

    # Add some noise for realism
    risk_score += np.random.normal(0, 0.05, len(df))
    risk_score = np.clip(risk_score, 0, 1)

    # Convert to health score (inverse of risk)
    df['health_score'] = np.round((1 - risk_score) * 100, 1)

    # Categorize
    conditions = [
        risk_score < 0.25,
        (risk_score >= 0.25) & (risk_score < 0.45),
        (risk_score >= 0.45) & (risk_score < 0.65),
        risk_score >= 0.65
    ]
    categories = [0, 1, 2, 3]  # LOW, MEDIUM, HIGH, CRITICAL
    df['risk_category'] = np.select(conditions, categories, default=1)

    df['risk_label'] = df['risk_category'].map({
        0: 'LOW', 1: 'MEDIUM', 2: 'HIGH', 3: 'CRITICAL'
    })

    return df


def get_feature_columns():
    """Return the list of feature columns used for model training."""
    return [
        'motor_age_years',
        'kw',
        'rpm',
        'voltage',
        'current',
        'duty_cycle_encoded',
        'make_encoded',
        'equipment_type_encoded',
        'application_use_encoded',
        'repair_count',
        'days_since_last_repair',
        'previous_refurb_count',
        'bearing_failure_count',
        'winding_failure_count',
        'rotor_failure_count',
        'insulation_failure_count',
        'dept_failure_rate',
        'make_failure_rate',
    ]


def run_feature_engineering(df, encoders=None, fit=True):
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Raw ERS DataFrame
        encoders: Pre-fitted label encoders (for inference)
        fit: Whether to fit encoders (True for training, False for inference)
    
    Returns:
        df: DataFrame with all engineered features
        encoders: Fitted label encoders
    """
    print("  [1/7] Computing motor age...")
    df = compute_motor_age(df)

    print("  [2/7] Computing repair history...")
    df = compute_repair_history(df)

    print("  [3/7] Computing failure history counts...")
    df = compute_failure_history(df)

    print("  [4/7] Computing department failure rate...")
    df = compute_department_failure_rate(df)

    print("  [5/7] Computing manufacturer failure rate...")
    df = compute_make_failure_rate(df)

    print("  [6/7] Encoding categorical features...")
    df, encoders = encode_categoricals(df, encoders=encoders, fit=fit)

    print("  [7/7] Creating risk target variable...")
    df = create_risk_target(df)

    return df, encoders


def main():
    """Run feature engineering on dummy data."""
    print("=" * 70)
    print("  JSW Steel ERS - Feature Engineering Pipeline")
    print("=" * 70)

    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dummy_ers_data.csv")
    if not os.path.exists(data_path):
        print(f"\n  ERROR: Data file not found at {data_path}")
        print("  Run generate_dummy_data.py first!")
        return

    print(f"\n  Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} records")

    # Run pipeline
    print(f"\n  Running feature engineering pipeline...")
    df, encoders = run_feature_engineering(df, fit=True)

    # Save engineered data
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    engineered_path = os.path.join(output_dir, "engineered_ers_data.csv")
    df.to_csv(engineered_path, index=False)
    print(f"\n  Saved engineered data to: {engineered_path}")

    # Save encoders
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    encoders_path = os.path.join(models_dir, "label_encoders.pkl")
    joblib.dump(encoders, encoders_path)
    print(f"  Saved label encoders to: {encoders_path}")

    # Feature summary
    feature_cols = get_feature_columns()
    print(f"\n  Feature Summary ({len(feature_cols)} features):")
    print(f"  {'Feature':<30s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print(f"  {'-'*70}")
    for col in feature_cols:
        if col in df.columns:
            print(f"  {col:<30s} {df[col].mean():>10.2f} {df[col].std():>10.2f} "
                  f"{df[col].min():>10.2f} {df[col].max():>10.2f}")

    print(f"\n  Target Distribution (risk_category):")
    for label, count in df['risk_label'].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"    {label:10s}: {count:4d} ({pct:.1f}%)")

    print(f"\n  Health Score Statistics:")
    print(f"    Mean:   {df['health_score'].mean():.1f}")
    print(f"    Median: {df['health_score'].median():.1f}")
    print(f"    Std:    {df['health_score'].std():.1f}")
    print(f"    Min:    {df['health_score'].min():.1f}")
    print(f"    Max:    {df['health_score'].max():.1f}")

    print(f"\n{'=' * 70}")
    print(f"  Feature engineering complete!")
    print(f"{'=' * 70}")

    return df, encoders


if __name__ == "__main__":
    main()
