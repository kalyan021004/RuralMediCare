#!/usr/bin/env python3
# RuralMediCare - Sync ALL Patients from Realtime Database to Firestore

import firebase_admin
from firebase_admin import credentials, db, firestore
import time
from datetime import datetime

# ===== INITIALIZE FIREBASE =====
SERVICE_ACCOUNT_PATH = "ruralmedicare-7c398-firebase-adminsdk-fbsvc.json"

try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://ruralmedicare-7c398-default-rtdb.asia-southeast1.firebasedatabase.app'
    })
    print("[✓] Firebase initialized\n")
except Exception as e:
    print(f"[✗] Firebase init failed: {e}")
    exit(1)

rtdb_ref = db.reference('patients')  # Root reference to all patients
fs = firestore.client()

# ===== SYNC DATA FOR ONE PATIENT =====
def sync_patient_to_firestore(patient_id, vitals_data):
    """Sync one patient's data to Firestore"""
    
    if not vitals_data:
        return 0
    
    synced = 0
    
    for key, reading in vitals_data.items():
        try:
            # Prepare data
            data = {
                'heartRate': int(reading.get('heartRate', 0)),
                'spo2': int(reading.get('spo2', 97)),
                'temperature': float(reading.get('temperature', 36.8)),
                'ecgRaw': int(reading.get('ecgRaw', 512)),
                'count': int(reading.get('count', 0)),
                'patientID': reading.get('patientID', patient_id),
                'timestamp': datetime.now(),
                'source': 'arduino_r4'
            }
            
            # Write to Firestore: patients/{patient_id}/vitals/{auto_id}
            fs.collection('patients').document(patient_id).collection('vitals').add(data)
            
            synced += 1
            
        except Exception as e:
            print(f"    [✗] Error syncing reading: {e}")
    
    return synced

# ===== SYNC ALL PATIENTS =====
def sync_all_patients():
    """Read all patients from Realtime DB and sync to Firestore"""
    
    try:
        # Get ALL patients data from Realtime Database
        all_patients = rtdb_ref.get()
        
        if not all_patients:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] No patients in Realtime DB")
            return 0
        
        total_synced = 0
        patient_count = 0
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(all_patients)} patients")
        
        # Loop through each patient
        for patient_id, patient_data in all_patients.items():
            
            # Get vitals for this patient
            vitals_data = patient_data.get('vitals', {})
            
            if not vitals_data:
                continue
            
            readings_count = len(vitals_data)
            print(f"  [{patient_id}] {readings_count} readings")
            
            # Sync this patient's data
            synced = sync_patient_to_firestore(patient_id, vitals_data)
            
            if synced > 0:
                # Delete ONLY vitals from Realtime DB (keeps patient node)
                rtdb_ref.child(patient_id).child('vitals').delete()
                print(f"  [✓] Synced & deleted {synced} vitals for {patient_id}")
                total_synced += synced
                patient_count += 1
        
        if total_synced > 0:
            print(f"\n[✓] TOTAL: Synced {total_synced} readings from {patient_count} patients\n")
        else:
            print()
        
        return total_synced
        
    except Exception as e:
        print(f"[✗] Sync error: {e}\n")
        return 0

# ===== CONTINUOUS MONITORING =====
def monitor_and_sync():
    """Continuously monitor ALL patients and sync"""
    
    print("=" * 60)
    print("RuralMediCare - Multi-Patient Sync (RTDB → Firestore)")
    print("=" * 60 + "\n")
    print("Monitoring ALL patients continuously...\n")
    
    sync_count = 0
    
    while True:
        try:
            # Check for any patient data
            all_patients = rtdb_ref.get()
            
            if all_patients:
                # Count total readings across all patients
                total_readings = 0
                patient_list = []
                
                for patient_id, patient_data in all_patients.items():
                    vitals = patient_data.get('vitals', {})
                    if vitals:
                        count = len(vitals)
                        total_readings += count
                        patient_list.append(f"{patient_id}({count})")
                
                if total_readings > 0:
                    print(f"[NEW DATA] {total_readings} readings from: {', '.join(patient_list)}")
                    synced = sync_all_patients()
                    if synced > 0:
                        sync_count += 1
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for new data...")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No patients yet")
            
            # Check every 3 seconds
            time.sleep(3)
            
        except KeyboardInterrupt:
            print(f"\n[✓] Stopped after {sync_count} sync operations")
            break
        except Exception as e:
            print(f"[✗] Error: {e}")
            time.sleep(3)

# ===== MANUAL SYNC (ONE TIME) =====
def manual_sync_once():
    """Manually sync all patients once"""
    
    print("=" * 60)
    print("RuralMediCare - Manual Sync (ALL Patients)")
    print("=" * 60 + "\n")
    
    synced = sync_all_patients()
    
    if synced > 0:
        print(f"[✓] Successfully synced {synced} total readings")
    else:
        print("[!] No data to sync")

# ===== MAIN =====
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # One-time sync
        manual_sync_once()
    else:
        # Continuous monitoring (default)
        monitor_and_sync()