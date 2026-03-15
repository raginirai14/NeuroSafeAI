from datetime import datetime
import os

LOG_FILE = os.path.join(os.path.dirname(__file__), "alert_log.txt")

def send_alert(patient_id, risk_level, probability, action):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert = {
        "timestamp"  : timestamp,
        "patient_id" : patient_id,
        "risk_level" : risk_level,
        "probability": probability,
        "action"     : action,
        "alert_sent" : True
    }
    print("\n" + "="*55)
    print("🚨 NEUROSAFE AI — EMERGENCY ALERT")
    print("="*55)
    print(f"  ⏰ Time       : {timestamp}")
    print(f"  🏥 Patient ID : {patient_id}")
    print(f"  🔴 Risk Level : {risk_level}")
    print(f"  📊 Probability: {probability}%")
    print(f"  💊 Action     : {action}")
    print("="*55)
    return alert

def log_result(patient_id, risk_level, probability):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line  = f"{timestamp} | Patient: {patient_id} | Risk: {risk_level} | Probability: {probability}%\n"
    with open(LOG_FILE, "a") as f:
        f.write(log_line)
    print(f"📝 Logged: {log_line.strip()}")
