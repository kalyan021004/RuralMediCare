from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from lime.lime_tabular import LimeTabularExplainer
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)

# -------------------- FIREBASE INIT --------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("ruralmedicare-7c398-firebase-adminsdk-fbsvc.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -------------------- LOAD ML ARTIFACTS --------------------
model = pickle.load(open('model.pkl', 'rb'))
le_gender = pickle.load(open('gender_encoder.pkl', 'rb'))
le_risk = pickle.load(open('risk_encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # NEW: Load scaler
risk_mapping = pickle.load(open('risk_mapping.pkl', 'rb'))  # NEW: Load risk mapping

# LIME dataset prep
df = pd.read_csv('health_data.csv')
selected_features = ['Heart Rate', 'Body Temperature', 'Oxygen Saturation', 'Age', 'Gender']
df = df[selected_features + ['Risk Category']]
df = df.drop('Risk Category', axis=1)
df['Gender'] = le_gender.transform(df['Gender'])
train_data = df.values

# -------------------- VALIDATION --------------------
VALID_RANGES = {
    'Heart Rate': (30, 240),
    'Body Temperature': (27, 42),
    'Oxygen Saturation': (70, 100),
    'Age': (18, 89)
}

def get_category_ranges(hr, spo2, temp):
    """Categorize vitals into Low/Mid/High"""
    hr_cat = "Low" if hr < 60 else "Mid" if hr <= 100 else "High"
    spo2_cat = "Low" if spo2 < 95 else "Mid" if spo2 <= 100 else "High"
    temp_cat = "Low" if temp < 36 else "Mid" if temp <= 37.5 else "High"
    return hr_cat, spo2_cat, temp_cat

def get_risk_color_and_icon(risk_level):
    """Return color, icon, and alert level for each risk category"""
    risk_info = {
        'Normal': {
            'color': '#10B981',  # Green
            'bg_color': '#D1FAE5',
            'icon': '✓',
            'alert': 'success',
            'message': 'All vitals are within normal range'
        },
        'Caution': {
            'color': '#F59E0B',  # Yellow
            'bg_color': '#FEF3C7',
            'icon': '⚠',
            'alert': 'warning',
            'message': 'Minor deviation detected - monitor closely'
        },
        'Moderate Risk': {
            'color': '#F97316',  # Orange
            'bg_color': '#FFEDD5',
            'icon': '⚠',
            'alert': 'warning',
            'message': 'Requires medical attention soon'
        },
        'High Risk': {
            'color': '#EF4444',  # Red
            'bg_color': '#FEE2E2',
            'icon': '⚠',
            'alert': 'danger',
            'message': 'Urgent medical care needed'
        },
        'Critical': {
            'color': '#DC2626',  # Dark Red
            'bg_color': '#FEE2E2',
            'icon': '🚨',
            'alert': 'danger',
            'message': 'EMERGENCY - Immediate intervention required'
        }
    }
    return risk_info.get(risk_level, risk_info['Normal'])

# -------------------- FETCH VITALS FROM FIREBASE --------------------
def fetch_latest_vitals(patient_id):
    try:
        vitals_ref = db.collection("patients").document(patient_id).collection("vitals")
        vitals = vitals_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).get()

        if not vitals:
            return None, None, None

        hr_list, temp_list, spo2_list = [], [], []

        for v in vitals:
            d = v.to_dict()
            if "heartRate" in d: hr_list.append(float(d["heartRate"]))
            if "temperature" in d: temp_list.append(float(d["temperature"]))
            if "spo2" in d: spo2_list.append(float(d["spo2"]))

        hr = round(sum(hr_list) / len(hr_list), 2) if hr_list else None
        temp = round(sum(temp_list) / len(temp_list), 2) if temp_list else None
        spo2 = round(sum(spo2_list) / len(spo2_list), 2) if spo2_list else None

        return hr, temp, spo2

    except Exception as e:
        print("❌ Firestore Error:", e)
        return None, None, None

# -------------------- ROUTES --------------------
@app.route('/')
def home():
    patients = []
    docs = db.collection("patients").get()

    for doc in docs:
        data = doc.to_dict()

        # Fetch latest vitals to predict risk
        hr, temp, spo2 = fetch_latest_vitals(doc.id)

        if hr is None or temp is None or spo2 is None:
            risk_level = 'unknown'
        else:
            gender_str = data.get("gender", "Male")
            try:
                gender = le_gender.transform([gender_str])[0]
            except:
                gender = le_gender.transform(["Male"])[0]

            numeric_features = ['Heart Rate', 'Body Temperature', 'Oxygen Saturation', 'Age']
            input_data = pd.DataFrame([{
                'Heart Rate': hr,
                'Body Temperature': temp,
                'Oxygen Saturation': spo2,
                'Age': data.get("age", 45),
                'Gender': gender
            }])

            input_data_scaled = input_data.copy()
            input_data_scaled[numeric_features] = scaler.transform(input_data[numeric_features])

            risk_pred = model.predict(input_data_scaled)[0]
            risk_display = risk_mapping[risk_pred]

            # Map Risk Name to Color Class
            risk_class_map = {
                'Caution': 0,
                'High Risk': 1,
                'Moderate Risk': 2,
                'Normal': 3
            }
            risk_level = risk_class_map.get(risk_display, 'unknown')

        patients.append({
            "id": doc.id,
            "name": data.get("patient_name", "Unknown"),
            "age": data.get("age", "?"),
            "location": data.get("location", "Unknown"),
            "risk_level": risk_level
        })

    return render_template('home.html', patients=patients)

@app.route('/patient/<patient_id>')
def patient_details(patient_id):
    patient_doc = db.collection("patients").document(patient_id).get()
    if not patient_doc.exists:
        return f"❌ Patient {patient_id} not found"

    patient = patient_doc.to_dict()

    # Fetch latest vitals
    hr, temp, spo2 = fetch_latest_vitals(patient_id)

    # Check if vitals exist
    if hr is None or temp is None or spo2 is None:
        return render_template('patient.html', 
                             patient=patient, 
                             patient_id=patient_id,
                             error="No vital signs data available")

    # Get vital categories
    hr_cat, spo2_cat, temp_cat = get_category_ranges(hr, spo2, temp)

    # Get patient gender (use from database or default)
    gender_str = patient.get("gender", "Male")
    try:
        gender = le_gender.transform([gender_str])[0]
    except:
        gender = le_gender.transform(["Male"])[0]  # default

    # Prepare input for prediction
    numeric_features = ['Heart Rate', 'Body Temperature', 'Oxygen Saturation', 'Age']
    input_data = pd.DataFrame([{
        'Heart Rate': hr,
        'Body Temperature': temp,
        'Oxygen Saturation': spo2,
        'Age': patient.get("age", 45),
        'Gender': gender
    }])

    # IMPORTANT: Scale the numeric features before prediction
    input_data_scaled = input_data.copy()
    input_data_scaled[numeric_features] = scaler.transform(input_data[numeric_features])

    # Make prediction
    proba = model.predict_proba(input_data_scaled)[0]
    risk_pred = model.predict(input_data_scaled)[0]
    risk_display = risk_mapping[risk_pred]

    # Get confidence for predicted class
    confidence = proba[risk_pred]

    # Get risk styling
    risk_info = get_risk_color_and_icon(risk_display)

    # Get all class probabilities for detailed view
    all_probabilities = []
    for idx, prob in enumerate(proba):
        risk_name = risk_mapping[idx]
        all_probabilities.append({
            'name': risk_name,
            'probability': f"{prob:.1%}",
            'value': prob
        })
    all_probabilities.sort(key=lambda x: x['value'], reverse=True)

    return render_template(
        'patient.html',
        patient=patient,
        patient_id=patient_id,
        heart_rate=hr,
        spo2=spo2,
        temperature=temp,
        hr_cat=hr_cat,
        spo2_cat=spo2_cat,
        temp_cat=temp_cat,
        risk=risk_display,
        confidence=f"{confidence:.1%}",
        risk_color=risk_info['color'],
        risk_bg_color=risk_info['bg_color'],
        risk_icon=risk_info['icon'],
        risk_message=risk_info['message'],
        all_probabilities=all_probabilities
    )

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """API endpoint for risk prediction"""
    try:
        data = request.json
        
        # Prepare input
        numeric_features = ['Heart Rate', 'Body Temperature', 'Oxygen Saturation', 'Age']
        input_data = pd.DataFrame([{
            'Heart Rate': float(data.get('heartRate')),
            'Body Temperature': float(data.get('temperature')),
            'Oxygen Saturation': float(data.get('spo2')),
            'Age': int(data.get('age')),
            'Gender': le_gender.transform([data.get('gender', 'Male')])[0]
        }])
        
        # Scale features
        input_data_scaled = input_data.copy()
        input_data_scaled[numeric_features] = scaler.transform(input_data[numeric_features])
        
        # Predict
        proba = model.predict_proba(input_data_scaled)[0]
        risk_pred = model.predict(input_data_scaled)[0]
        risk_display = risk_mapping[risk_pred]
        
        # Get risk info
        risk_info = get_risk_color_and_icon(risk_display)
        
        return jsonify({
            'risk': risk_display,
            'confidence': float(proba[risk_pred]),
            'color': risk_info['color'],
            'message': risk_info['message'],
            'probabilities': {risk_mapping[i]: float(p) for i, p in enumerate(proba)}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("🚀 Starting Rural Health Dashboard with Multi-Level Risk Classification")
    print("\nRisk Levels:")
    for idx, risk in risk_mapping.items():
        print(f"  {idx}: {risk}")
    print("\n" + "="*60)
    app.run(debug=True, host='127.0.0.1', port=5000)