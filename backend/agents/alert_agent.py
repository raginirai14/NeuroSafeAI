"""
NeuroSafeAI — Agentic Alert System
This agent automatically monitors every prediction result
and fires alerts without any human clicking anything.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../alerts"))
from notifier import send_alert, log_result

# Risk thresholds
CRITICAL_THRESHOLD = 75.0
HIGH_THRESHOLD     = 60.0

def run_agent(patient_id, prediction_result):
    """
    The core agent logic.
    
    Takes prediction result and automatically decides:
    - Should an alert be fired?
    - What action should be taken?
    - Log the result

    Args:
        patient_id (str): unique patient identifier
        prediction_result (dict): result from AI model
            {
                "hemorrhage_probability": 82.4,
                "risk_level": "Critical",
                "action": "IMMEDIATE intervention needed",
                "color": "red"
            }

    Returns:
        dict: agent response with alert status
    """
    probability = prediction_result["hemorrhage_probability"]
    risk_level  = prediction_result["risk_level"]
    action      = prediction_result["action"]

    print(f"\n🤖 Agent monitoring patient {patient_id}...")
    print(f"   Probability : {probability}%")
    print(f"   Risk Level  : {risk_level}")

    # Always log every result
    log_result(patient_id, risk_level, probability)

    # Agent decision logic
    alert_fired = False
    agent_response = {
        "patient_id"  : patient_id,
        "probability" : probability,
        "risk_level"  : risk_level,
        "alert_fired" : False,
        "message"     : ""
    }

    if probability >= CRITICAL_THRESHOLD:
        print(f"🚨 CRITICAL DETECTED — Agent firing alert automatically!")
        send_alert(patient_id, risk_level, probability, action)
        agent_response["alert_fired"] = True
        agent_response["message"]     = "🚨 Emergency alert sent automatically"

    elif probability >= HIGH_THRESHOLD:
        print(f"⚠️  HIGH RISK DETECTED — Agent sending urgent notification!")
        send_alert(patient_id, risk_level, probability, action)
        agent_response["alert_fired"] = True
        agent_response["message"]     = "⚠️ Urgent notification sent"

    else:
        print(f"✅ Risk acceptable — No alert needed")
        agent_response["message"] = "✅ No alert required"

    return agent_response
