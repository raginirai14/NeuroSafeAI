# 🧠 NeuroSafeAI

> *A doctor uploads a CT scan. 2 seconds later — the AI already knows.*

---

## the idea

Brain hemorrhage kills people because it gets caught too late.

NeuroSafeAI is an end-to-end AI system that looks at CT scan images, detects hemorrhage using deep learning, and autonomously alerts medical staff — no human in the loop.

This isn't a tutorial project. Every component was built from scratch.

---

## what makes this hard to build

most people can't do this because it requires knowing:

- how to train a deep learning model on real medical data
- how to build and serve a REST API that connects to that model
- how to design an autonomous agent that makes decisions without human input
- how to wire a full frontend to a live backend
- how to handle real CT scan images end to end

this project does all five.

---

## how it works
```
CT scan uploaded via dashboard
           ↓
FastAPI backend receives the image
           ↓
EfficientNet-B0 model analyzes pixel patterns
           ↓
Returns hemorrhage probability (0–100%)
           ↓
Autonomous agent evaluates the score
           ↓
≥75% → emergency alert fires automatically
           ↓
Result logged with timestamp, shown on dashboard
```

---

## the stack

| layer | technology | why |
|-------|-----------|-----|
| AI Model | PyTorch + EfficientNet-B0 | state of the art image classification |
| Training | Transfer Learning + Adam | fast convergence on small dataset |
| Backend | FastAPI + Uvicorn | async, fast, production ready |
| Agent | Custom Python | autonomous decision making |
| Frontend | HTML + CSS + JS | lightweight, no framework needed |

---

## model performance

trained on 200 real CT scan images from Kaggle
```
Training Accuracy   →  97.14%
Validation Accuracy →  83.33%
Epochs              →  20
Loss Function       →  Binary Cross Entropy
Optimizer           →  Adam (lr=0.0001)
Architecture        →  EfficientNet-B0 + custom classifier head
```

---

## the agentic part

this is not just a model with an API.

there is an autonomous agent running on every prediction that:
- evaluates risk without being asked
- decides whether to fire an alert based on thresholds
- logs every decision with a timestamp
- acts immediately on critical cases

this is what separates it from a basic ML project.

---

## risk classification
```
0  – 30%   🟢  Low Risk       no action needed
30 – 60%   🟡  Moderate       monitor the patient
60 – 80%   🟠  High Risk      urgent evaluation
80 – 100%  🔴  Critical       autonomous alert fired 🚨
```

---

## run it locally
```bash
# clone
git clone https://github.com/raginirai14/NeuroSafeAI.git
cd NeuroSafeAI

# setup
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision fastapi uvicorn pillow python-multipart

# start server
cd backend
uvicorn main:app --port 8080

# open dashboard (new terminal)
open frontend/dashboard/login.html
```

login with `doctor` / `neuro123` or create your own account.

---

## project structure
```
NeuroSafeAI/
├── ai_models/
│   └── ct_scan_model/
│       ├── model.py          EfficientNet-B0 architecture
│       ├── train.py          training pipeline
│       ├── predict.py        inference engine
│       └── checkpoints/      saved weights
├── backend/
│   ├── main.py               FastAPI application
│   └── agents/
│       └── alert_agent.py    autonomous monitoring agent
├── frontend/
│   └── dashboard/
│       ├── index.html        medical dashboard
│       └── login.html        auth system
└── alerts/
    ├── notifier.py           alert dispatcher
    └── alert_log.txt         audit trail
```

---

## what's next

- [ ] DICOM support (actual hospital format)
- [ ] email + SMS alerts
- [ ] larger dataset for higher accuracy
- [ ] cloud deployment
- [ ] multi-class hemorrhage detection

---

*built end to end by Ragini Rai*
*deep learning · agentic AI · full stack*
