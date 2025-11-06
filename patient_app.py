from flask import Flask, render_template, request, redirect
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import requests

app = Flask(__name__)

if not firebase_admin._apps:
    cred = credentials.Certificate("ruralmedicare-7c398-firebase-adminsdk-fbsvc.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

def get_category_ranges(hr, spo2, temp):
    hr_cat = "Low" if hr < 60 else "Mid" if hr <= 100 else "High"
    spo2_cat = "Low" if spo2 < 95 else "Mid" if spo2 <= 100 else "High"
    temp_cat = "Low" if temp < 36 else "Mid" if temp <= 37.5 else "High"
    return hr_cat, spo2_cat, temp_cat

def latest_vitals(pid):
    v = db.collection("patients").document(pid).collection("vitals") \
        .order_by("timestamp", direction=firestore.Query.DESCENDING).limit(3).get()

    if not v:
        return None, None, None

    hr = np.mean([float(x.to_dict().get("heartRate", np.nan)) for x in v])
    spo2 = np.mean([float(x.to_dict().get("spo2", np.nan)) for x in v])
    temp = np.mean([float(x.to_dict().get("temperature", np.nan)) for x in v])
    return round(hr,2), round(spo2,2), round(temp,2)

@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        return redirect(f"/view/{request.form['patient_id'].upper()}")
    return render_template("patient_access.html")

@app.route("/view/<pid>")
def view(pid):
    p = db.collection("patients").document(pid).get()
    if not p.exists:
        return "Patient Not Found"

    patient = p.to_dict()
    hr, spo2, temp = latest_vitals(pid)
    if hr is None:
        return "No vitals yet."

    payload = {
        "heartRate": hr,
        "temperature": temp,
        "spo2": spo2,
        "age": patient.get("age",40),
        "gender": patient.get("gender","Male")
    }

    res = requests.post("http://127.0.0.1:5000/api/predict", json=payload).json()

    hr_cat, spo2_cat, temp_cat = get_category_ranges(hr, spo2, temp)

    return render_template("patient_view.html",
    patient=patient,
    heart_rate=hr,
    spo2=spo2,
    temperature=temp,
    hr_cat=hr_cat,
    spo2_cat=spo2_cat,
    temp_cat=temp_cat,
    risk=res["risk"],
    confidence=f"{res['confidence']:.1%}"
)


if __name__ == "__main__":
    app.run(port=7000, debug=True)
