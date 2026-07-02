import os
import sys
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from predict import load_model_and_encoders, predict_single_motor, batch_predict
from feature_engineering import get_feature_columns

# Initialize FastAPI
app = FastAPI(title="JSW ERS Predictive Maintenance API")

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Variables for the model and data
model = None
encoders = None
feature_cols = None
latest_records = None
raw_df = None

# Helper function to recursively convert numpy types to native python types
import numpy as np
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

@app.on_event("startup")
def startup_event():
    global model, encoders, feature_cols, latest_records, raw_df
    print("Loading models and data...")
    try:
        model, encoders = load_model_and_encoders(MODELS_DIR)
        feature_cols = get_feature_columns()
        
        # Load raw data
        raw_df = pd.read_csv(os.path.join(DATA_DIR, "dummy_ers_data.csv"))
        
        # Load baseline data records
        df = pd.read_csv(os.path.join(DATA_DIR, "engineered_ers_data.csv"))
        # Retrieve latest status per equipment
        latest_records = df.sort_values('received_date').groupby('equipment_serial_no').last().reset_index()
        print(f"Loaded {len(latest_records)} unique motors into memory.")
    except Exception as e:
        print(f"Error loading models or data: {str(e)}")

@app.get("/")
async def root():
    """Serve the main frontend HTML file."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/api/motors")
async def get_all_motors():
    """Return a list of all motor IDs for the dropdown."""
    if latest_records is None:
        return {"error": "Data not loaded"}
    serials = latest_records['equipment_serial_no'].tolist()
    return {"motors": serials}

@app.get("/api/predict/{serial_no}")
async def get_prediction(serial_no: str):
    """Predict risk for a specific motor."""
    if latest_records is None or model is None:
        return {"error": "System not fully initialized"}
        
    motor_row = latest_records[latest_records['equipment_serial_no'] == serial_no]
    if motor_row.empty:
        return {"error": f"Motor {serial_no} not found"}
        
    motor_series = motor_row.iloc[0]
    
    # Generate prediction
    pred = predict_single_motor(model, motor_series, feature_cols)
    
    # Combine with basic motor metadata for the UI
    response = {
        "metadata": {
            "serial_no": str(serial_no),
            "equipment_type": str(motor_series.get('equipment_type_text', 'N/A')),
            "manufacturer": str(motor_series.get('make', 'N/A')),
            "installed_location": str(motor_series.get('installed_location', 'N/A')),
            "kw_rating": motor_series.get('kw', 0),
            "rpm": motor_series.get('rpm', 0),
            "age_years": motor_series.get('motor_age_years', 0)
        },
        "prediction": pred
    }
    
    return convert_numpy(response)

class CustomPredictRequest(BaseModel):
    serial_no: str
    checklist: List[str]

@app.post("/api/predict/custom")
async def get_custom_prediction(request: CustomPredictRequest):
    if latest_records is None or model is None:
        return {"error": "System not fully initialized"}
        
    motor_row = latest_records[latest_records['equipment_serial_no'] == request.serial_no]
    if motor_row.empty:
        return {"error": f"Motor {request.serial_no} not found"}
        
    # Copy features to modify
    motor_series = motor_row.iloc[0].copy()
    
    # Apply checklist modifiers
    for item in request.checklist:
        if item == "R2" or item == "G6":  # Bearing Seating Condition / Loose Fasteners
            motor_series['bearing_failure_count'] += 1
        elif item == "S1" or item == "S3":  # Stator Winding Condition / Burning Marks
            motor_series['winding_failure_count'] += 1
        elif item == "R3" or item == "R1":  # Rotor Bar Damage / Rotor Shaft Damage
            motor_series['rotor_failure_count'] += 1
        elif item == "S2":  # Insulation Damage
            motor_series['insulation_failure_count'] += 1
            
        # General modifiers for other items
        if item in ["G4", "G5", "R5", "S5"]:
            motor_series['repair_count'] += 1
            motor_series['previous_refurb_count'] += 1
            motor_series['days_since_last_repair'] = min(motor_series['days_since_last_repair'], 14)
            
    # Generate prediction using modified series
    pred = predict_single_motor(model, motor_series, feature_cols)
    
    response = {
        "metadata": {
            "serial_no": str(request.serial_no),
            "equipment_type": str(motor_series.get('equipment_type_text', 'N/A')),
            "manufacturer": str(motor_series.get('make', 'N/A')),
            "installed_location": str(motor_series.get('installed_location', 'N/A')),
            "kw_rating": motor_series.get('kw', 0),
            "rpm": motor_series.get('rpm', 0),
            "age_years": motor_series.get('motor_age_years', 0)
        },
        "prediction": pred
    }
    
    return convert_numpy(response)

@app.get("/api/motors/registry")
async def get_motor_registry():
    if latest_records is None or model is None:
        return {"error": "System not fully initialized"}
        
    # Evaluate current fleet status
    batch_df = batch_predict(model, latest_records, feature_cols)
    
    records = []
    for idx, row in batch_df.iterrows():
        records.append({
            "equipment_id": int(row.get('requisition_id', idx + 1)),
            "allotment": "dr" if idx % 3 == 0 else "",  # Assign allotment status
            "main_equipment_name": str(row.get('equipment_type_text', 'N/A')),
            "equipment_serial_no": str(row['equipment_serial_no']),
            "motor_manufacturing_year": int(row.get('motor_manufacturing_year', 2020)),
            "make": str(row.get('make', 'N/A')),
            "model": str(row.get('model', 'N/A')),
            "kw": float(row.get('kw', 0)),
            "frame": str(row.get('frame', 'N/A')),
            "rpm": int(row.get('rpm', 0)),
            "voltage": int(row.get('voltage', 0)),
            "installed_location": str(row.get('installed_location', 'N/A')),
            "status": str(row.get('status', 'N/A')),
            "health_score": float(row['health_score']),
            "predicted_risk": str(row['predicted_risk'])
        })
        
    return convert_numpy(records)

def compute_dashboard_stats():
    if raw_df is None:
        return {}
    
    yet_to_start = int(raw_df['status'].eq('RECEIVED').sum())
    work_up = int(raw_df['status'].isin(['UNDER_INSPECTION', 'UNDER_REPAIR', 'TESTING']).sum())
    ers_pending = yet_to_start + work_up
    
    user_pending = int(raw_df['status'].eq('DISPATCHED').sum())
    repaired_closed = int(raw_df['status'].eq('CLOSED').sum())
    
    dispatched_df = raw_df[raw_df['status'] == 'DISPATCHED']
    bearings_req = int(dispatched_df['service_type_text'].str.contains('Bearing', case=False, na=False).sum())
    spares_req = int(dispatched_df['service_type_text'].str.contains('Rewinding|Insulation', case=False, na=False).sum())
    mech_jobs = len(dispatched_df) - bearings_req - spares_req
    
    in_repair_count = ers_pending
    repaired_count = user_pending + repaired_closed
    
    return {
        "pending_status": {
            "ers": ers_pending,
            "user_dept": user_pending
        },
        "pending_ers": {
            "yet_to_start": yet_to_start,
            "work_up": work_up
        },
        "pending_user": {
            # Segment pending items by repair classification
            "mechanical_jobs": mech_jobs,
            "spares_required": spares_req
        },
        "lt_motors": {
            "received": in_repair_count,
            "repaired": repaired_count
        }
    }

def compute_category_stats():
    if raw_df is None:
        return []
    
    categories = [
        {"name": "> 100 KW", "query": raw_df['kw'] > 100},
        {"name": "25.1 - 100 KW", "query": (raw_df['kw'] > 25) & (raw_df['kw'] <= 100)},
        {"name": "<= 25 KW", "query": raw_df['kw'] <= 25}
    ]
    
    results = []
    for cat in categories:
        sub_df = raw_df[cat["query"]]
        
        received_outstanding = int(sub_df['status'].ne('CLOSED').sum())
        repaired = int(sub_df['status'].isin(['CLOSED', 'DISPATCHED']).sum())
        at_ers = int(sub_df['status'].isin(['RECEIVED', 'UNDER_INSPECTION', 'UNDER_REPAIR', 'TESTING']).sum())
        
        results.append({
            "category": cat["name"],
            "received_outstanding": received_outstanding,
            "repaired": repaired,
            "at_ers": at_ers
        })
        
    return results

def compute_turnaround_stats():
    if raw_df is None:
        return {}
        
    df_closed = raw_df[raw_df['status'].isin(['CLOSED', 'DISPATCHED'])].copy()
    df_closed['received_dt'] = pd.to_datetime(df_closed['received_date'])
    df_closed['updated_dt'] = pd.to_datetime(df_closed['updated_at'])
    df_closed['duration_days'] = (df_closed['updated_dt'] - df_closed['received_dt']).dt.days.fillna(0)
    
    rewinding_df = df_closed[df_closed['service_type_text'].str.contains('Rewinding', case=False, na=False)]
    overhauling_df = df_closed[df_closed['service_type_text'].str.contains('Overhaul|Overhauling', case=False, na=False)]
    
    rew_0_7 = int((rewinding_df['duration_days'] <= 7).sum())
    rew_8_15 = int(((rewinding_df['duration_days'] > 7) & (rewinding_df['duration_days'] <= 15)).sum())
    rew_15_30 = int(((rewinding_df['duration_days'] > 15) & (rewinding_df['duration_days'] <= 30)).sum())
    rew_30_plus = int((rewinding_df['duration_days'] > 30).sum())
    rew_total = rew_0_7 + rew_8_15 + rew_15_30 + rew_30_plus
    
    ovh_0_3 = int((overhauling_df['duration_days'] <= 3).sum())
    ovh_4_7 = int(((overhauling_df['duration_days'] > 3) & (overhauling_df['duration_days'] <= 7)).sum())
    ovh_7_15 = int(((overhauling_df['duration_days'] > 7) & (overhauling_df['duration_days'] <= 15)).sum())
    ovh_15_plus = int((overhauling_df['duration_days'] > 15).sum())
    ovh_total = ovh_0_3 + ovh_4_7 + ovh_7_15 + ovh_15_plus
    
    return {
        "rewinding": {
            "0_7": rew_0_7,
            "8_15": rew_8_15,
            "15_30": rew_15_30,
            "30_plus": rew_30_plus,
            "total": rew_total
        },
        "overhauling": {
            "0_3": ovh_0_3,
            "4_7": ovh_4_7,
            "7_15": ovh_7_15,
            "15_plus": ovh_15_plus,
            "total": ovh_total
        }
    }

@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    if raw_df is None:
        return {"error": "Data not loaded"}
        
    return {
        "dashboard": compute_dashboard_stats(),
        "categories": compute_category_stats(),
        "turnaround": compute_turnaround_stats()
    }

@app.get("/api/fleet")
async def get_fleet_summary():
    """Return a summary of the entire fleet."""
    if latest_records is None or model is None:
        return {"error": "System not fully initialized"}
        
    batch_df = batch_predict(model, latest_records, feature_cols)
    
    risk_counts = batch_df['predicted_risk'].value_counts().to_dict()
    
    for r in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        if r not in risk_counts:
            risk_counts[r] = 0
            
    top_critical = batch_df.nlargest(5, 'risk_class')
    top_motors = []
    
    for _, row in top_critical.iterrows():
        top_motors.append({
            "serial_no": row['equipment_serial_no'],
            "type": row.get('equipment_type_text', 'N/A'),
            "location": row.get('installed_location', 'N/A'),
            "health_score": float(row['health_score']),
            "risk": row['predicted_risk']
        })
        
    total_motors = len(batch_df)
    avg_health = float(batch_df['health_score'].mean())
    
    response = {
        "total_motors": total_motors,
        "avg_health_score": round(avg_health, 1),
        "risk_distribution": risk_counts,
        "top_at_risk": top_motors
    }
    
    return convert_numpy(response)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
