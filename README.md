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
