from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../ai_models/ct_scan_model"))
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))

from predict import predict
from alert_agent import run_agent

app = FastAPI(title="NeuroSafeAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def root():
    return {"status": "NeuroSafeAI API is running ✅"}

@app.post("/predict")
async def predict_scan(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run AI model
    result = predict(
        image_path=temp_path,
        model_path=os.path.join(os.path.dirname(__file__), "../ai_models/ct_scan_model/checkpoints/best_model.pth")
    )

    # Run agent — automatically fires alert if critical
    patient_id     = file.filename.replace(".png", "").replace(".jpg", "")
    agent_response = run_agent(patient_id, result)

    # Clean up temp file
    os.remove(temp_path)

    # Return combined result
    return {**result, "agent": agent_response}
