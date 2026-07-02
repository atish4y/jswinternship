"""
JSW Steel ERS — Synthetic Data Generator
=========================================
Generates realistic dummy data mimicking JSW Steel's Electrical Repair Shop
requisition records for training predictive maintenance models.

Features:
- 2000+ records with realistic distributions
- Correlated failure patterns (older motors + higher KW → higher risk)
- Multiple departments, equipment types, and manufacturers
- Historical repair chains for the same motors
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================================
# CONFIGURATION - Realistic JSW Steel ERS Parameters
# ============================================================================

DEPARTMENTS = [
    "Hot Strip Mill", "Cold Rolling Mill", "Coke Oven Plant",
    "Blast Furnace", "Sinter Plant", "Power Plant",
    "Wire Rod Mill", "Bar Mill"
]

EQUIPMENT_TYPES = [
    ("AC Motor", 0.45), ("DC Motor", 0.15), ("Pump Motor", 0.15),
    ("Blower Motor", 0.10), ("Crane Motor", 0.08), ("Compressor Motor", 0.07)
]

MANUFACTURERS = [
    ("ABB", 0.18), ("Siemens", 0.16), ("Crompton Greaves", 0.14),
    ("Kirloskar", 0.12), ("Bharat Bijlee", 0.10), ("WEG", 0.08),
    ("Havells", 0.07), ("GE", 0.06), ("Toshiba", 0.05), ("Remi", 0.04)
]

FAILURE_REASONS = {
    "Bearing Failure":     {"id": 1, "weight": 0.40},
    "Winding Failure":     {"id": 2, "weight": 0.25},
    "Rotor Failure":       {"id": 3, "weight": 0.15},
    "Insulation Failure":  {"id": 4, "weight": 0.12},
    "Overheating":         {"id": 5, "weight": 0.05},
    "Vibration Damage":    {"id": 6, "weight": 0.03},
}

SERVICE_TYPES = [
    ("Rewinding", 1, 0.30), ("Bearing Replacement", 2, 0.25),
    ("General Overhaul", 3, 0.20), ("Alignment Correction", 4, 0.10),
    ("Insulation Repair", 5, 0.08), ("Shaft Repair", 6, 0.07)
]

BEARING_TYPES = [
    ("6205 2RS", 1, 0.15), ("6206 2RS", 2, 0.12), ("6305 2RS", 3, 0.10),
    ("6306 2RS", 4, 0.10), ("6307 2RS", 5, 0.08), ("6308 2RS", 6, 0.08),
    ("NU 206", 7, 0.07), ("NU 308", 8, 0.07), ("7206 B", 9, 0.06),
    ("22210 E", 10, 0.05), ("N/A", 11, 0.12)
]

DUTY_CYCLES = [
    ("Continuous (S1)", 0.55), ("Short Time (S2)", 0.15),
    ("Intermittent (S3)", 0.15), ("Continuous with Intermittent Load (S6)", 0.10),
    ("Periodic (S5)", 0.05)
]

APPLICATION_USES = [
    "Rolling Mill Drive", "Pump Driving", "Fan/Blower Driving",
    "Crane Hoisting", "Compressor Driving", "Conveyor Belt",
    "Cooling Tower Fan", "Hydraulic Press", "Mixer Agitator"
]

STATUSES = ["RECEIVED", "UNDER_INSPECTION", "UNDER_REPAIR", "TESTING", "DISPATCHED", "CLOSED"]

FRAMES = ["D80", "D90L", "D100L", "D112M", "D132S", "D132M", "D160M", "D160L",
           "D180M", "D180L", "D200L", "D225S", "D225M", "D250M", "D280S", "D280M",
           "D315S", "D315M", "D355M", "D355L"]


def weighted_choice(items_with_weights):
    """Select item from list of (item, weight) tuples."""
    items = [x[0] for x in items_with_weights]
    weights = [x[1] for x in items_with_weights]
    return np.random.choice(items, p=weights)


def generate_motor_specs():
    """Generate realistic motor specifications."""
    kw_values = [0.75, 1.1, 1.5, 2.2, 3.7, 5.5, 7.5, 11, 15, 18.5, 22, 30, 37, 45,
                 55, 75, 90, 110, 132, 160, 200, 250, 315, 400, 500]
    kw_probs = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.08, 0.07, 0.05, 0.06,
                         0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02,
                         0.01, 0.01, 0.01])
    kw_probs = kw_probs / kw_probs.sum()  # Normalize to exactly 1.0
    kw = np.random.choice(kw_values, p=kw_probs)

    # Voltage depends on KW
    if kw <= 7.5:
        voltage = np.random.choice([415, 440], p=[0.7, 0.3])
    elif kw <= 132:
        voltage = np.random.choice([415, 440, 3300], p=[0.5, 0.3, 0.2])
    else:
        voltage = np.random.choice([3300, 6600, 11000], p=[0.4, 0.4, 0.2])

    # RPM
    rpm = np.random.choice([750, 1000, 1500, 3000], p=[0.10, 0.15, 0.50, 0.25])

    # Current derived from KW and voltage (approximate)
    power_factor = np.random.uniform(0.82, 0.92)
    if voltage <= 440:
        current = (kw * 1000) / (np.sqrt(3) * voltage * power_factor)
    else:
        current = (kw * 1000) / (np.sqrt(3) * voltage * power_factor)
    current = round(current * np.random.uniform(0.95, 1.05), 2)

    # KVA
    kva = round(kw / power_factor, 2)

    # Power rating ~ KW (sometimes slightly different)
    power_rating = round(kw * np.random.uniform(0.98, 1.02), 2)

    return kw, voltage, rpm, current, kva, power_rating


def generate_records(n_motors=500, n_records=2000):
    """
    Generate n_records requisition records for n_motors unique motors.
    Motors with more repairs get higher failure risk (realistic correlation).
    """
    records = []

    # First, generate a pool of unique motors
    motors = {}
    for i in range(n_motors):
        serial = f"EQ{10000 + i:05d}"
        mfg_year = np.random.choice(
            range(1995, 2024),
            p=np.array([max(0.01, 0.08 - 0.002 * abs(y - 2010)) for y in range(1995, 2024)]) /
              sum([max(0.01, 0.08 - 0.002 * abs(y - 2010)) for y in range(1995, 2024)])
        )
        make_tuple = [(m, w) for m, w in MANUFACTURERS]
        make = weighted_choice(make_tuple)
        eq_type_tuple = [(t, w) for t, w, *_ in EQUIPMENT_TYPES]
        eq_type = weighted_choice(eq_type_tuple)
        dept = np.random.choice(DEPARTMENTS)
        app_use = np.random.choice(APPLICATION_USES)
        kw, voltage, rpm, current, kva, power_rating = generate_motor_specs()
        duty = weighted_choice(DUTY_CYCLES)
        frame = np.random.choice(FRAMES)

        motors[serial] = {
            "make": make,
            "equipment_type": eq_type,
            "mfg_year": int(mfg_year),
            "department": dept,
            "application_use": app_use,
            "kw": kw,
            "voltage": voltage,
            "rpm": rpm,
            "current": current,
            "kva": kva,
            "power_rating": power_rating,
            "duty_cycle": duty,
            "frame": frame,
            "repair_count": 0,
        }

    # Generate requisition records
    # Some motors will have multiple repairs (realistic pattern)
    req_id = 1
    for record_idx in range(n_records):
        # Bias motor selection: some motors fail more often
        if record_idx < n_motors:
            # First pass: one record per motor
            serial = f"EQ{10000 + record_idx:05d}"
        else:
            # Subsequent: weighted towards older/larger motors (more likely to fail again)
            weights = []
            serials = list(motors.keys())
            for s in serials:
                m = motors[s]
                age = 2026 - m["mfg_year"]
                w = (age / 30) * 0.4 + (m["kw"] / 500) * 0.3 + (m["repair_count"] / max(1, n_records // n_motors)) * 0.3
                weights.append(max(0.001, w))
            weights = np.array(weights) / sum(weights)
            serial = np.random.choice(serials, p=weights)

        motor = motors[serial]
        motor["repair_count"] += 1

        # Motor age affects failure probability
        motor_age = 2026 - motor["mfg_year"]

        # === FAILURE REASON (correlated with age and kw) ===
        failure_items = list(FAILURE_REASONS.keys())
        failure_weights = [FAILURE_REASONS[f]["weight"] for f in failure_items]

        # Older motors → more bearing/insulation failures
        if motor_age > 15:
            failure_weights[0] *= 1.3  # Bearing
            failure_weights[3] *= 1.5  # Insulation
        if motor["kw"] > 100:
            failure_weights[1] *= 1.2  # Winding
            failure_weights[2] *= 1.3  # Rotor

        failure_weights = np.array(failure_weights) / sum(failure_weights)
        failure_reason = np.random.choice(failure_items, p=failure_weights)
        failure_id = FAILURE_REASONS[failure_reason]["id"]

        # === SERVICE TYPE (correlated with failure reason) ===
        service_items = [(s[0], s[1], s[2]) for s in SERVICE_TYPES]
        if "Bearing" in failure_reason:
            service_text, service_id = "Bearing Replacement", 2
        elif "Winding" in failure_reason:
            service_text, service_id = "Rewinding", 1
        elif "Insulation" in failure_reason:
            service_text, service_id = "Insulation Repair", 5
        else:
            st = service_items[np.random.randint(len(service_items))]
            service_text, service_id = st[0], st[1]

        # === BEARING TYPE ===
        bearing_tuple = [(b[0], b[2]) for b in BEARING_TYPES]
        bearing_text = weighted_choice(bearing_tuple)
        bearing_id = next(b[1] for b in BEARING_TYPES if b[0] == bearing_text)

        # === DATES ===
        # Spread records over 3 years
        base_date = datetime(2023, 1, 1)
        received_date = base_date + timedelta(days=np.random.randint(0, 1095))
        received_time = f"{np.random.randint(6, 22):02d}:{np.random.randint(0, 60):02d}:00"

        # === WARRANTY ===
        warranty = "Under Warranty" if motor_age < 3 and np.random.random() < 0.7 else "Out of Warranty"

        # === PREVIOUS REFURBISHMENT ===
        prev_refurb = ""
        if motor["repair_count"] > 1 and np.random.random() < 0.6:
            prev_refurb = f"JOB-{np.random.randint(10000, 99999)}"

        # === DEFECTS and REPAIRS ===
        defect_templates = {
            "Bearing Failure": [
                "Bearing noise observed during operation",
                "Excessive vibration from drive end bearing",
                "Bearing seized, motor stalled",
                "Grease leakage from bearing housing",
                "Bearing overheating detected"
            ],
            "Winding Failure": [
                "Insulation resistance low on winding",
                "Phase imbalance detected",
                "Winding burnt, motor tripped on overload",
                "Inter-turn short circuit detected",
                "Winding overheated, discoloration visible"
            ],
            "Rotor Failure": [
                "Rotor bars broken, abnormal sound",
                "Rotor rubbing against stator",
                "Air gap irregularity detected",
                "Shaft bent, causing vibration",
                "Rotor eccentricity observed"
            ],
            "Insulation Failure": [
                "Low megger value on phases",
                "Insulation breakdown at connections",
                "Surface tracking on insulation",
                "Moisture ingress caused insulation degradation",
                "Insulation resistance below acceptable limit"
            ],
            "Overheating": [
                "Motor overheating during normal load",
                "Temperature rise exceeding limits",
                "Cooling fan damaged, inadequate cooling"
            ],
            "Vibration Damage": [
                "Excessive vibration at both ends",
                "Foundation bolt loose causing vibration",
                "Coupling misalignment detected"
            ]
        }

        repair_templates = {
            "Bearing Failure": "Bearings replaced and housing cleaned. Shaft journal inspected. Alignment checked.",
            "Winding Failure": "Complete rewinding done. New insulation applied. Impregnation and baking completed.",
            "Rotor Failure": "Rotor repaired/replaced. Dynamic balancing done. Air gap checked.",
            "Insulation Failure": "Insulation restored. Varnish applied and baked. Megger values confirmed acceptable.",
            "Overheating": "Cooling system repaired. Fan replaced. Ventilation passages cleaned.",
            "Vibration Damage": "Alignment corrected. Coupling replaced. Foundation bolts tightened."
        }

        defects = np.random.choice(defect_templates.get(failure_reason, ["General defect observed"]))
        repairs = repair_templates.get(failure_reason, "General repair performed")

        # === STATUS (time-based) ===
        days_ago = (datetime(2026, 6, 20) - received_date).days
        if days_ago > 60:
            status = np.random.choice(["DISPATCHED", "CLOSED"], p=[0.3, 0.7])
        elif days_ago > 30:
            status = np.random.choice(["UNDER_REPAIR", "TESTING", "DISPATCHED", "CLOSED"], p=[0.1, 0.15, 0.35, 0.4])
        elif days_ago > 14:
            status = np.random.choice(["UNDER_INSPECTION", "UNDER_REPAIR", "TESTING", "DISPATCHED"], p=[0.1, 0.3, 0.3, 0.3])
        else:
            status = np.random.choice(["RECEIVED", "UNDER_INSPECTION", "UNDER_REPAIR"], p=[0.3, 0.4, 0.3])

        # === MODEL NAME ===
        model_prefixes = ["M", "K", "E", "D", "H", "S"]
        model_name = f"{np.random.choice(model_prefixes)}{np.random.randint(100, 999)}"

        # === SPARES ===
        spares = []
        if "Bearing" in failure_reason or "Bearing" in service_text:
            spares.append(f"Bearing {bearing_text}")
        if "Winding" in failure_reason or "Rewinding" in service_text:
            spares.append("Copper wire, Insulation paper, Varnish")
        if not spares:
            spares.append("Grease, Cleaning agents")
        spares_text = "; ".join(spares)

        # === EQUIPMENT TYPE ID ===
        eq_type_id = next((i + 1 for i, (t, _) in enumerate(EQUIPMENT_TYPES) if t == motor["equipment_type"]), 1)

        record = {
            "requisition_id": req_id,
            "main_equipment_name": motor["equipment_type"],
            "equipment_serial_no": serial,
            "motor_manufacturing_year": motor["mfg_year"],
            "application_use": motor["application_use"],
            "installed_location": motor["department"],
            "installed_qty": 1,
            "make": motor["make"],
            "model": model_name,
            "frame": motor["frame"],
            "size": f'{motor["kw"]} KW',
            "power_rating": motor["power_rating"],
            "kw": motor["kw"],
            "kva": motor["kva"],
            "rpm": motor["rpm"],
            "duty_cycle": motor["duty_cycle"],
            "voltage": motor["voltage"],
            "current": motor["current"],
            "received_date": received_date.strftime("%Y-%m-%d"),
            "received_time": received_time,
            "defects_observed": defects,
            "repairs_required": repairs,
            "reason_of_failure_id": failure_id,
            "reason_of_failure_text": failure_reason,
            "warranty_status": warranty,
            "previous_refurbishment_job_no": prev_refurb,
            "spares_details": spares_text,
            "service_type_id": service_id,
            "service_type_text": service_text,
            "equipment_type_id": eq_type_id,
            "equipment_type_text": motor["equipment_type"],
            "bearing_id": bearing_id,
            "bearing_text": bearing_text,
            "status": status,
            "created_at": received_date.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": (received_date + timedelta(days=np.random.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S"),
            "created_by": f"operator_{np.random.randint(1, 20):02d}",
            "updated_by": f"supervisor_{np.random.randint(1, 8):02d}",
        }

        records.append(record)
        req_id += 1

    return pd.DataFrame(records)


def main():
    print("=" * 70)
    print("  JSW Steel ERS — Synthetic Data Generator")
    print("=" * 70)

    # Create output directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    # Generate data
    print("\n[1/3] Generating 2000 synthetic ERS records for 500 motors...")
    df = generate_records(n_motors=500, n_records=2000)

    # Save to CSV
    output_path = os.path.join(data_dir, "dummy_ers_data.csv")
    df.to_csv(output_path, index=False)
    print(f"[2/3] Saved to: {output_path}")

    # Summary statistics
    print(f"\n[3/3] Dataset Summary:")
    print(f"  Total Records:         {len(df)}")
    print(f"  Unique Motors:         {df['equipment_serial_no'].nunique()}")
    print(f"  Date Range:            {df['received_date'].min()} to {df['received_date'].max()}")
    print(f"  Departments:           {df['installed_location'].nunique()}")
    print(f"  Equipment Types:       {df['equipment_type_text'].nunique()}")
    print(f"  Manufacturers:         {df['make'].nunique()}")

    print(f"\n  Failure Distribution:")
    for reason, count in df['reason_of_failure_text'].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {reason:25s}: {count:4d} ({pct:.1f}%)")

    print(f"\n  Department Distribution:")
    for dept, count in df['installed_location'].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {dept:25s}: {count:4d} ({pct:.1f}%)")

    print(f"\n  Status Distribution:")
    for status, count in df['status'].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {status:25s}: {count:4d} ({pct:.1f}%)")

    print(f"\n{'=' * 70}")
    print(f"  Data generation complete!")
    print(f"{'=' * 70}")

    return df


if __name__ == "__main__":
    main()
