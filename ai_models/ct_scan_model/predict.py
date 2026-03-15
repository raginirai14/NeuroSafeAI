import torch
from torchvision import transforms
from PIL import Image
import sys, os
sys.path.append(os.path.dirname(__file__))
from model import load_trained_model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        ),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def classify_risk(prob):
    if prob < 0.30:
        return {"risk_level": "Low Risk",      "action": "No immediate action needed",   "color": "green"}
    elif prob < 0.60:
        return {"risk_level": "Moderate Risk", "action": "Monitor patient closely",       "color": "yellow"}
    elif prob < 0.80:
        return {"risk_level": "High Risk",     "action": "Urgent evaluation required",    "color": "orange"}
    else:
        return {"risk_level": "Critical",      "action": "IMMEDIATE intervention needed", "color": "red"}

def predict(image_path, model_path="checkpoints/best_model.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_trained_model(model_path, device)
    tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        prob = model(tensor).item()
    return {"hemorrhage_probability": round(prob * 100, 2), **classify_risk(prob)}
