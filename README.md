# ClusterHealth AI

**Predictive & Explainable GPU/CPU Cluster Management**

AI Innovation Hackathon 2026 — Department of CSE, Daffodil International University
Track: *AI for Cluster Intelligence (Predictive Operations)*
Team: **NephroAI**

A hybrid XGBoost + BiGRU-Attention system that predicts GPU/CPU hardware failures
before they happen, explains every prediction with SHAP, and gives operators a
live dashboard with an adjustable, data-driven alert threshold.

---

## ✨ Key Features

- **Hybrid failure prediction** — XGBoost (tabular) + BiGRU-Attention (sequence),
  combined as a weighted ensemble (`0.6 × XGBoost + 0.4 × BiGRU`)
- **SHAP explainability** — every prediction shows *why* a device was flagged
- **Report Card & Threshold Tuner** *(Final Round challenge module)* — lets an
  operator see and adjust the trade-off between missed faults and false alarms,
  live, on stored data — with an auto-suggested optimal alert level
- **Live demo mode** — an 8-device simulated cluster that visibly drifts from
  healthy → warning → high risk over a live session
- **Cost Optimizer** — flags underutilized GPUs and converts idle time into
  real daily/monthly dollar waste estimates
- **Kubernetes-aware action** — simulated (or real, if connected) `kubectl cordon`
  + `kubectl drain` workflow triggered from the dashboard
- **Live alerting** — Slack / Discord / Email notifications on risk escalation
- **Full observability stack** — Prometheus + Grafana, auto-provisioned

---

## 🏗️ Architecture

```
Telemetry (simulated / real Prometheus & DCGM)
        │
        ▼
Feature Engineering (sliding window, 10 timesteps)
        │
   ┌────┴────┐
   ▼         ▼
XGBoost   BiGRU-Attention
   │         │
   └────┬────┘
        ▼
  Weighted Ensemble (0.6 / 0.4)
        │
        ▼
  SHAP Explainability
        │
        ▼
FastAPI (inference + Report Card + alerts + k8s actions)
        │
        ▼
Dashboard (live risk, cost optimizer, report card tuner)
```

---

## 📁 Project Structure

```
clusterhealth-ai/
├── src/
│   ├── generate_data.py           # synthetic telemetry generator
│   ├── features.py                # feature engineering / windowing
│   ├── train_xgboost.py           # tabular failure classifier
│   ├── train_bigru_attention.py   # sequential deep learning model
│   ├── ensemble_and_explain.py    # weighted ensemble + SHAP
│   ├── api.py                     # FastAPI inference service
│   ├── report_card.py             # Report Card & Threshold Tuner (Final Round)
│   ├── metrics_exporter.py        # simulates a live cluster (Prometheus)
│   ├── risk_exporter.py           # republishes AI risk scores as metrics
│   ├── k8s_action.py              # cordon/drain automation
│   └── alerts.py                  # Slack / Discord / Email alerting
├── dashboard/
│   └── index.html                 # live dashboard (no build step)
├── docker/
│   ├── docker-compose.yml         # one-command full stack
│   ├── Dockerfile.api / .exporters / .dashboard
│   ├── prometheus/                # Prometheus scrape config
│   └── grafana/                   # auto-provisioned dashboards
├── k8s/
│   └── deployment.yaml            # reference production K8s manifest
├── docs/
│   └── DESIGN_NOTE_report_card.md
├── models/                        # trained model artifacts
├── data/
│   └── telemetry.csv              # synthetic labeled telemetry
└── requirements.txt
```

---

## 🚀 Quick Start (Docker — recommended)

```bash
cd docker
cp .env.example .env      # set GRAFANA_ADMIN_PASSWORD (required)
docker compose up --build
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:8080 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

## 🐍 Manual Setup (without Docker)

```bash
pip install -r requirements.txt
cd src
python3 generate_data.py
python3 train_xgboost.py
python3 train_bigru_attention.py
python3 ensemble_and_explain.py
uvicorn api:app --reload --port 8000
```
Then open `dashboard/index.html` in a browser.

---

## 📋 Report Card & Threshold Tuner

The alert threshold used to be fixed in code — too strict and real faults get
missed, too loose and the operator drowns in false alarms. This module makes
that trade-off visible and adjustable, without retraining anything.

- Every labeled window in `data/telemetry.csv` is scored **once** at startup
  by the same trained ensemble, then cached — so moving the threshold slider
  is instant (a NumPy comparison, not a re-inference call)
- **Report Card**: live counts of *Faults Caught*, *Faults Missed*, *False Alarms*
- **Threshold Tuner**: a slider (5%–95%) + Low/Medium/High presets
- **Suggested Level**: sweeps the full threshold range and recommends the level
  that maximizes an F₂ score (recall-weighted, since a missed fault costs more
  than one extra false-alarm check), with a reason generated from real numbers

```bash
curl "http://localhost:8000/reportcard/config"
curl "http://localhost:8000/reportcard?threshold=0.40"
curl "http://localhost:8000/reportcard/suggested"
```

Full design rationale: [`docs/DESIGN_NOTE_report_card.md`](docs/DESIGN_NOTE_report_card.md)

---

## 🧠 ML Pipeline

1. **Data** — 300 simulated devices, 96 timesteps each, 12 sensor features
   (GPU temp, utilization, power draw, ECC errors, etc.), ~18% injected with
   realistic failure signatures
2. **Windowing** — 10-timestep sliding windows feed both models
3. **XGBoost** — gradient-boosted trees on tabular features (SHAP-compatible)
4. **BiGRU-Attention** — bidirectional GRU + attention over raw sequences
5. **Ensemble** — `0.6 × XGBoost + 0.4 × BiGRU`
6. **SHAP** — per-prediction feature attribution for explainability

> **Note:** telemetry is synthetically generated for the hackathon (no
> production cluster access), modeled after real DCGM failure signatures.
> The entire pipeline accepts live Prometheus/DCGM data with zero model
> code changes.

---

## 🛠️ Tech Stack

`XGBoost` · `PyTorch (BiGRU-Attention)` · `SHAP` · `FastAPI` · `Prometheus` ·
`Grafana` · `Docker Compose` · `Kubernetes`

---

## 👥 Team

**Team NephroAI** — AI Innovation Hackathon 2026, DIU CSE
