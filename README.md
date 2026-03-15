# 🧠 NeuroSafeAI
> *"What if an AI could spot a brain bleed before a doctor even opens the file?"*

That's exactly what this does.

---

## ⚡ what is this thing

A full-stack AI system that looks at CT scan images of brains and goes —
**"yeah that's bleeding"** or **"you're fine"** — in seconds.

No clicking. No waiting. No manual review.
Just upload → analyze → alert. Done.

---

## 🔥 what makes it cool
```
you upload a brain scan
         ↓
an AI model stares at it really hard
         ↓
spits out a probability of hemorrhage
         ↓
an autonomous agent wakes up
         ↓
if it's bad → SCREAMS an emergency alert
if it's fine → quietly logs it and goes back to sleep
         ↓
doctor sees everything on a clean dashboard
```

no human needed in the middle. that's the point.

---

## 📊 the numbers

| thing | number |
|-------|--------|
| training accuracy | 97% |
| validation accuracy | 83% |
| CT scans trained on | 200 |
| epochs | 20 |
| time to analyze | ~2 seconds |

---

## 🛠️ built with
```
brain power  →  PyTorch + EfficientNet-B0
server       →  FastAPI + Uvicorn
agent        →  pure Python chaos
dashboard    →  HTML + CSS + JS
vibes        →  10/10
```

---

## 🚀 run it yourself
```bash
git clone https://github.com/raginirai14/NeuroSafeAI.git
cd NeuroSafeAI
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision fastapi uvicorn pillow python-multipart
cd backend
uvicorn main:app --port 8080
open frontend/dashboard/login.html
```

---

## 🎯 risk levels

| probability | verdict |
|-------------|---------|
| 0 - 30% | 🟢 you're probably fine |
| 30 - 60% | 🟡 keep an eye on it |
| 60 - 80% | 🟠 okay get checked |
| 80 - 100% | 🔴 🚨🚨�� |

---

*built from scratch. no shortcuts. just code, CT scans, and a lot of terminal errors.*

**— Ragini Rai** 🧠
