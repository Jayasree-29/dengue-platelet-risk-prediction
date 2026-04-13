from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
import json
from typing import Optional, List, Dict

app = FastAPI(title="Dengue Platelet Guardian API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models and Artifacts
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
RANGES_PATH = os.path.join(os.path.dirname(__file__), 'feature_ranges.json')

feature_ranges = {}
if os.path.exists(RANGES_PATH):
    with open(RANGES_PATH, 'r') as f:
        feature_ranges = json.load(f)

try:
    lr_model = joblib.load(os.path.join(MODELS_DIR, 'lr_model.pkl'))
    rf_model = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
    xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl'))
    reg_model = joblib.load(os.path.join(MODELS_DIR, 'reg_model.pkl'))
    
    imputer = joblib.load(os.path.join(MODELS_DIR, 'imputer.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    le_gender = joblib.load(os.path.join(MODELS_DIR, 'le_gender.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    reg_feature_names = joblib.load(os.path.join(MODELS_DIR, 'reg_feature_names.pkl'))
    scale_cols = joblib.load(os.path.join(MODELS_DIR, 'scale_cols.pkl'))
    
    print("All models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    pass

class PatientData(BaseModel):
    Age: float
    Gender: str
    Hemoglobin: float
    Neutrophils: float
    Lymphocytes: float
    Monocytes: float
    Eosinophils: float
    RBC: float
    HCT: float
    MCV: float
    MCH: float
    MCHC: float
    RDW_CV: float
    Total_Platelet_Count: float
    MPV: float
    PDW: float
    PCT: float
    Total_WBC_count: float
    Previous_Platelet_Count: Optional[float] = None
    Time_Since_Last_Test_Hours: Optional[float] = 24.0 

class ValidationWarning(BaseModel):
    field: str
    value: float
    range: str

class PredictionResponse(BaseModel):
    Risk_Level: str
    Dengue_Probability: float
    Dengue_Result_Prediction: str
    Decline_Forecast: float
    Clinical_Alert_Flag: str
    Decision_Support: str
    Dengue_Likelihood_Level: str
    Feature_Importance: dict
    Validation_Warnings: List[ValidationWarning]

@app.get("/")
def home():
    return {"message": "Dengue Platelet Guardian API is running"}

@app.get("/ranges")
def get_ranges():
    return feature_ranges

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    try:
        input_data = data.dict()
        
        # 1. Validation Check (before normalization)
        validation_warnings = []
        
        # Field mapping for validation (Pydantic field -> CSV field name)
        valid_map = {
            'Age': 'Age',
            'Hemoglobin': 'Hemoglobin',
            'Neutrophils': 'Neutrophils',
            'Lymphocytes': 'Lymphocytes',
            'Monocytes': 'Monocytes',
            'Eosinophils': 'Eosinophils',
            'RBC': 'RBC',
            'HCT': 'HCT',
            'MCV': 'MCV',
            'MCH': 'MCH',
            'MCHC': 'MCHC',
            'RDW_CV': 'RDW-CV',
            'Total_Platelet_Count': 'Total Platelet Count',
            'MPV': 'MPV',
            'PDW': 'PDW',
            'PCT': 'PCT',
            'Total_WBC_count': 'Total WBC count'
        }

        for py_field, csv_field in valid_map.items():
            if csv_field in feature_ranges:
                val = input_data[py_field]
                meta = feature_ranges[csv_field]
                
                # Check for UI-scaled counts (10^3) during validation
                check_val = val
                if csv_field in ['Total Platelet Count', 'Total WBC count'] and val < 1000:
                    check_val = val * 1000
                
                if check_val < meta['min'] or check_val > meta['max']:
                    validation_warnings.append({
                        "field": csv_field,
                        "value": float(val),
                        "range": f"{meta['min']} - {meta['max']}"
                    })

        # 2. Prepare Data (Normalization)
        if input_data['Total_Platelet_Count'] < 1000:
            input_data['Total_Platelet_Count'] *= 1000
        if input_data['Previous_Platelet_Count'] and input_data['Previous_Platelet_Count'] < 1000:
            input_data['Previous_Platelet_Count'] *= 1000
        if input_data['Total_WBC_count'] < 100:
            input_data['Total_WBC_count'] *= 1000
            
        col_map = {
            'Hemoglobin': 'Hemoglobin',
            'Neutrophils': 'Neutrophils',
            'Lymphocytes': 'Lymphocytes',
            'Monocytes': 'Monocytes',
            'Eosinophils': 'Eosinophils',
            'HCT': 'HCT',
            'MCV': 'MCV',
            'MCH': 'MCH',
            'MCHC': 'MCHC',
            'RDW_CV': 'RDW-CV',
            'Total_Platelet_Count': 'Total Platelet Count',
            'MPV': 'MPV',
            'PDW': 'PDW',
            'PCT': 'PCT',
            'Total_WBC_count': 'Total WBC count'
        }
        
        df_input = pd.DataFrame([input_data])
        df_input.rename(columns=col_map, inplace=True)
        
        gender_str = df_input['Gender'].iloc[0]
        try:
            df_input['Gender'] = le_gender.transform([gender_str])[0]
        except:
            df_input['Gender'] = 0 if gender_str.lower() == 'male' else 1
            
        if scale_cols:
            try:
                subset_to_scale = df_input[scale_cols].copy()
                scaled_array = scaler.transform(subset_to_scale)
                df_scaled_subset = pd.DataFrame(scaled_array, columns=scale_cols, index=df_input.index)
                for col in scale_cols:
                    df_input[col] = df_scaled_subset[col]
            except Exception as e:
                print(f"Scaling error: {e}")
        
        df_model = df_input[feature_names].copy()
            
        # 3. Model Inference
        dengue_prob = rf_model.predict_proba(df_model)[0][1]
        
        reg_input = df_input[reg_feature_names].copy()
        forecast_val = reg_model.predict(reg_input)[0]
        
        platelet_col = 'Total Platelet Count'
        platelet_std = np.sqrt(scaler.var_[scale_cols.index(platelet_col)])
        platelet_mean = scaler.mean_[scale_cols.index(platelet_col)]
        forecast_platelet_count = (forecast_val * platelet_std) + platelet_mean
        
        # 4. Clinical Logic
        current_platelet = input_data['Total_Platelet_Count']
        
        if dengue_prob > 0.7:
            risk_score = 3
        elif dengue_prob > 0.4:
            risk_score = 2
        else:
            risk_score = 1
            
        alert_flag = "No"
        if current_platelet < 20000:
            risk_score = 3
            alert_flag = "Yes (Critical Thrombocytopenia)"
        elif current_platelet < 50000:
            risk_score = 3
            alert_flag = "Yes"
            
        if data.Previous_Platelet_Count:
            prev_p = data.Previous_Platelet_Count
            if prev_p < 1000: prev_p *= 1000
            drop = prev_p - current_platelet
            rate = drop / data.Time_Since_Last_Test_Hours
            if rate > 2000 and current_platelet < 100000:
                risk_score = max(risk_score, 3)
                alert_flag = "Yes (Rapid Decline)"
        
        # 5. Final Result Formation
        final_prob = dengue_prob * 100
        if risk_score == 3 and final_prob < 70:
            final_prob = 75.0 + (dengue_prob * 10)
        elif risk_score == 2 and (final_prob < 40 or final_prob > 70):
            final_prob = 55.0

        if final_prob >= 70:
            risk_level, dengue_likelihood_level = "High Risk", "High Likelihood"
            dengue_result_str = "High Risk (Likely Dengue)"
        elif final_prob >= 50:
            risk_level, dengue_likelihood_level = "Moderate Risk", "Moderate Likelihood"
            dengue_result_str = "Moderate Risk (Probable)"
        elif final_prob >= 40:
            risk_level, dengue_likelihood_level = "Low-Moderate Risk", "Moderate Likelihood"
            dengue_result_str = "Low-Moderate Risk (Suspected)"
        else:
            risk_level, dengue_likelihood_level = "Low Risk", "Low Likelihood"
            dengue_result_str = "Low Risk (Unlikely)"

        decision = "Immediate follow-up required." if risk_level == "High Risk" else \
                   "Close monitoring advised." if risk_level == "Moderate Risk" else \
                   "Routine observation sufficient."

        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_features = {feature_names[indices[f]]: float(importances[indices[f]]) for f in range(min(5, len(feature_names)))}

        return {
            "Risk_Level": risk_level,
            "Dengue_Probability": float(round(final_prob, 2)),
            "Dengue_Result_Prediction": dengue_result_str,
            "Dengue_Likelihood_Level": dengue_likelihood_level,
            "Decline_Forecast": float(round(forecast_platelet_count, 0)),
            "Clinical_Alert_Flag": alert_flag,
            "Decision_Support": decision,
            "Feature_Importance": top_features,
            "Validation_Warnings": validation_warnings
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
