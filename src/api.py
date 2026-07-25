"""Production-safe FastAPI inference service for ClusterHealth AI."""

from __future__ import annotations

import hmac
import os
import random
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

import alerts
import k8s_action
import live_data
import report_card
from features import RAW_FEATURES, WINDOW, tabular_feature_names
from model_service import HybridModelService, ModelServiceError


BASE = Path(__file__).resolve().parent.parent
MODELS = BASE / "models"
DATA_PATH = BASE / "data" / "telemetry.csv"
W_XGB, W_BIGRU = 0.6, 0.4
FEATURE_NAMES = tabular_feature_names()
DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
PREDICTION_COUNT = Counter("clusterhealth_predictions_total", "Predictions served", ["source"])
PREDICTION_LATENCY = Histogram("clusterhealth_prediction_seconds", "Prediction latency", ["source"])


def _cors_origins() -> list[str]:
    configured = os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080")
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.models = HybridModelService(MODELS)
        app.state.telemetry = pd.read_csv(DATA_PATH)
    except Exception as exc:
        raise RuntimeError(f"ClusterHealth API startup failed: {exc}") from exc
    # Warm the Report Card & Threshold Tuner cache once at boot (inference
    # only, over the already-generated stored dataset -- no training happens
    # here). This keeps the first slider move in a live demo instant instead
    # of paying the one-time scoring cost in front of the judges.
    try:
        app.state.report_card_cache = report_card.build_report_card_cache(app.state.models, app.state.telemetry)
    except report_card.ReportCardUnavailable as exc:
        app.state.report_card_cache = None
        print(f"[startup warning] Report Card cache not ready: {exc}")
    try:
        yield
    finally:
        app.state.models.close()


app = FastAPI(title="ClusterHealth AI API", version="1.0.0", lifespan=lifespan)
app.include_router(report_card.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Action-Token"],
)


class TelemetryWindow(BaseModel):
    device_id: int
    readings: list[list[float]] = Field(..., description=f"{WINDOW} readings of {len(RAW_FEATURES)} features")


def _models(request: Request) -> HybridModelService:
    return request.app.state.models


def _telemetry(request: Request) -> pd.DataFrame:
    return request.app.state.telemetry


def _tabular(sequence: np.ndarray) -> np.ndarray:
    if sequence.shape != (WINDOW, len(RAW_FEATURES)):
        raise ValueError(f"Expected readings shaped ({WINDOW}, {len(RAW_FEATURES)}), got {sequence.shape}")
    if not np.isfinite(sequence).all():
        raise ValueError("Readings must contain only finite numeric values")
    mean, std, last = sequence.mean(axis=0), sequence.std(axis=0), sequence[-1]
    slope = (sequence[-1] - sequence[0]) / WINDOW
    return np.concatenate([mean, std, last, slope]).astype(np.float32)


def recommend_action(probability: float, reasons: list[tuple[str, float]]) -> str:
    drivers = ", ".join(name for name, _ in reasons[:2]) or "no dominant driver"
    if probability >= 0.75:
        return f"URGENT: Drain and cordon this node now. Primary drivers: {drivers}."
    if probability >= 0.4:
        return f"WARNING: Schedule maintenance within 2h; avoid new long-running jobs. Watch: {drivers}."
    return "Healthy: no action needed."


def _risk_level(probability: float) -> str:
    return "high" if probability >= 0.75 else "medium" if probability >= 0.4 else "low"


def _score_batch(models: HybridModelService, sequences: np.ndarray, tabular: np.ndarray, *, explain: bool):
    try:
        return models.score(sequences, tabular, explain=explain)
    except (ModelServiceError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Inference unavailable: {exc}") from exc


def _format_prediction(xgb: float, bigru: float, shap_row: np.ndarray | None, attention: np.ndarray | None):
    probability = float(W_XGB * xgb + W_BIGRU * bigru)
    reasons: list[tuple[str, float]] = []
    if shap_row is not None:
        reasons = sorted(zip(FEATURE_NAMES, shap_row.tolist()), key=lambda item: -abs(item[1]))[:5]
        reasons = [(name, round(float(value), 4)) for name, value in reasons]
    return {
        "xgb_probability": round(float(xgb), 4),
        "bigru_probability": round(float(bigru), 4),
        "ensemble_probability": round(probability, 4),
        "risk_level": _risk_level(probability),
        "top_shap_reasons": reasons,
        "recommended_action": recommend_action(probability, reasons),
        "attention_weights": [] if attention is None else [round(float(value), 4) for value in attention],
    }


def _score_one(models: HybridModelService, sequence: np.ndarray):
    tab_row = _tabular(sequence)
    xgb, bigru, shap_values, attention = _score_batch(models, sequence[None, ...], tab_row[None, ...], explain=True)
    return _format_prediction(xgb[0], bigru[0], shap_values[0], attention[0])


def _validate_live_device(device_id: str) -> str:
    if not DEVICE_ID_RE.fullmatch(device_id):
        raise HTTPException(status_code=422, detail="Invalid device ID")
    return device_id


@app.get("/")
def root(request: Request):
    return {"status": "ok", "service": "ClusterHealth AI", "xgb_worker_alive": request.app.state.models.xgb.is_healthy}


@app.get("/healthz")
def healthz(request: Request):
    if not request.app.state.models.xgb.is_healthy:
        raise HTTPException(status_code=503, detail="XGBoost/SHAP worker is unavailable")
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/devices")
def list_devices(request: Request):
    return sorted(_telemetry(request)["device_id"].unique().tolist())


@app.post("/predict")
def predict_custom(payload: TelemetryWindow, request: Request, background_tasks: BackgroundTasks):
    sequence = np.asarray(payload.readings, dtype=np.float32)
    try:
        with PREDICTION_LATENCY.labels("custom").time():
            result = _score_one(_models(request), sequence)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    PREDICTION_COUNT.labels("custom").inc()
    result["device_id"] = payload.device_id
    if result["risk_level"] != "low":
        background_tasks.add_task(alerts.maybe_alert, payload.device_id, result["risk_level"], result["ensemble_probability"], result["top_shap_reasons"], result["recommended_action"])
    return result


@app.get("/predict/device/{device_id}")
def predict_for_device(device_id: int, request: Request, background_tasks: BackgroundTasks):
    frame = _telemetry(request)
    rows = frame[frame.device_id == device_id].sort_values("timestep")
    if len(rows) < WINDOW:
        raise HTTPException(status_code=404, detail=f"Not enough data for device {device_id}")
    window = rows.tail(WINDOW)
    sequence = window[RAW_FEATURES].to_numpy(dtype=np.float32)
    with PREDICTION_LATENCY.labels("dataset").time():
        result = _score_one(_models(request), sequence)
    PREDICTION_COUNT.labels("dataset").inc()
    result.update(device_id=device_id, last_timestep=int(window["timestep"].iloc[-1]), raw_last_reading={name: float(window[name].iloc[-1]) for name in RAW_FEATURES})
    if result["risk_level"] != "low":
        background_tasks.add_task(alerts.maybe_alert, device_id, result["risk_level"], result["ensemble_probability"], result["top_shap_reasons"], result["recommended_action"])
    return result


@app.get("/cluster/overview")
def cluster_overview(request: Request, background_tasks: BackgroundTasks):
    windows = []
    device_ids = []
    for device_id, group in _telemetry(request).groupby("device_id"):
        group = group.sort_values("timestep")
        if len(group) >= WINDOW:
            windows.append(group.tail(WINDOW)[RAW_FEATURES].to_numpy(dtype=np.float32))
            device_ids.append(int(device_id))
    if not windows:
        return {"total_devices": 0, "high_risk": 0, "medium_risk": 0, "devices": []}
    sequences = np.stack(windows)
    tabular = np.stack([_tabular(window) for window in sequences])
    with PREDICTION_LATENCY.labels("dataset_batch").time():
        xgb, bigru, _, _ = _score_batch(_models(request), sequences, tabular, explain=False)
    PREDICTION_COUNT.labels("dataset_batch").inc(len(device_ids))
    results = []
    alert_indices = []
    for index, (device_id, xgb_probability, bigru_probability) in enumerate(zip(device_ids, xgb, bigru)):
        result = _format_prediction(xgb_probability, bigru_probability, None, None)
        results.append({"device_id": device_id, "ensemble_probability": result["ensemble_probability"], "risk_level": result["risk_level"]})
        if result["risk_level"] != "low":
            alert_indices.append(index)
    if alert_indices:
        try:
            _, shap_values = _models(request).xgb.predict(tabular[alert_indices], explain=True)
        except ModelServiceError as exc:
            raise HTTPException(status_code=503, detail=f"SHAP explanation unavailable: {exc}") from exc
        for index, shap_row in zip(alert_indices, shap_values):
            probability = results[index]["ensemble_probability"]
            reasons = sorted(zip(FEATURE_NAMES, shap_row.tolist()), key=lambda item: -abs(item[1]))[:5]
            reasons = [(name, round(float(value), 4)) for name, value in reasons]
            background_tasks.add_task(alerts.maybe_alert, device_ids[index], results[index]["risk_level"], probability, reasons, recommend_action(probability, reasons))
    results.sort(key=lambda item: -item["ensemble_probability"])
    return {"total_devices": len(results), "high_risk": sum(item["risk_level"] == "high" for item in results), "medium_risk": sum(item["risk_level"] == "medium" for item in results), "devices": results}


# ---- Cost Optimizer -------------------------------------------------------
# Flags GPUs running under 40% utilization (the standard cloud FinOps
# consolidation threshold -- almost nothing in this dataset is *literally*
# idle at 0-15%) and converts idle time into power + opportunity-cost dollars.
POWER_COST_PER_KWH = 0.12
GPU_CLOUD_RENTAL_HOURLY = 3.50
IDLE_UTIL_THRESHOLD = 40.0


@app.get("/cost/analysis")
def cost_analysis(request: Request):
    frame = _telemetry(request)
    rows = []
    for device_id, group in frame.groupby("device_id"):
        latest = group.sort_values("timestep").tail(1).iloc[0]
        util = float(latest["gpu_util_pct"])
        power_w = float(latest["power_draw_w"])
        is_idle = util < IDLE_UTIL_THRESHOLD
        hourly_power_cost = (power_w / 1000.0) * POWER_COST_PER_KWH
        hourly_opportunity_cost = GPU_CLOUD_RENTAL_HOURLY if is_idle else 0.0
        rows.append({
            "device_id": int(device_id),
            "utilization_pct": round(util, 1),
            "power_draw_w": round(power_w, 1),
            "is_idle": bool(is_idle),
            "daily_waste_usd": round((hourly_power_cost + hourly_opportunity_cost) * 24, 2) if is_idle else 0.0,
        })
    idle_rows = [r for r in rows if r["is_idle"]]
    total_daily_waste = sum(r["daily_waste_usd"] for r in idle_rows)
    rows.sort(key=lambda r: -r["daily_waste_usd"])
    return {
        "total_devices": len(rows),
        "idle_devices": len(idle_rows),
        "idle_threshold_pct": IDLE_UTIL_THRESHOLD,
        "daily_waste_usd": round(total_daily_waste, 2),
        "monthly_waste_usd": round(total_daily_waste * 30, 2),
        "devices": rows,
    }


@app.get("/live/status")
def live_status():
    up = live_data.prometheus_up()
    return {"prometheus_reachable": up, "prometheus_url": live_data.PROMETHEUS_URL, "live_devices": live_data.list_live_devices() if up else []}


@app.get("/live/predict/device/{device_id}")
def live_predict_for_device(device_id: str, request: Request, background_tasks: BackgroundTasks):
    device_id = _validate_live_device(device_id)
    if not live_data.prometheus_up():
        raise HTTPException(status_code=503, detail="Prometheus is unavailable")
    try:
        sequence = live_data.build_live_window(device_id)
    except live_data.LiveDataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    with PREDICTION_LATENCY.labels("live").time():
        result = _score_one(_models(request), sequence)
    PREDICTION_COUNT.labels("live").inc()
    result.update(device_id=device_id, source="live_prometheus")
    if result["risk_level"] != "low":
        background_tasks.add_task(alerts.maybe_alert, hash(device_id), result["risk_level"], result["ensemble_probability"], result["top_shap_reasons"], result["recommended_action"])
    return result


@app.get("/live/cluster/overview")
def live_cluster_overview(request: Request):
    if not live_data.prometheus_up():
        raise HTTPException(status_code=503, detail="Prometheus is unavailable")
    devices = []
    for device_id in live_data.list_live_devices():
        try:
            sequence = live_data.build_live_window(device_id)
            result = _score_one(_models(request), sequence)
            devices.append({"device_id": device_id, "ensemble_probability": result["ensemble_probability"], "risk_level": result["risk_level"]})
        except (live_data.LiveDataUnavailable, HTTPException):
            continue
    devices.sort(key=lambda item: -item["ensemble_probability"])
    return {"total_devices": len(devices), "high_risk": sum(item["risk_level"] == "high" for item in devices), "medium_risk": sum(item["risk_level"] == "medium" for item in devices), "devices": devices}


_SIM_STATE: dict[int, dict] = {}


def _init_sim_state(frame: pd.DataFrame) -> None:
    if _SIM_STATE:
        return
    for index, device_id in enumerate(sorted(frame["device_id"].unique().tolist())[:8]):
        _SIM_STATE[int(device_id)] = {"failing": index in (2, 5), "tick": 0, "gpu_temp_c": random.uniform(58, 68), "gpu_util_pct": random.uniform(45, 85), "power_draw_w": random.uniform(190, 260), "fan_speed_pct": random.uniform(35, 55), "ecc_sbe_errors": 0, "ecc_dbe_errors": 0}


@app.get("/live/snapshot")
def live_snapshot(request: Request, background_tasks: BackgroundTasks):
    _init_sim_state(_telemetry(request))
    sequences, ids, raw_readings = [], [], []
    for device_id, state in _SIM_STATE.items():
        state["tick"] += 1
        ramp = min(1.0, state["tick"] / 25) if state["failing"] else 0.0
        state["gpu_temp_c"] = float(np.clip(state["gpu_temp_c"] + random.gauss(0.2 * ramp, 0.4), 40, 105))
        state["gpu_util_pct"] = float(np.clip(state["gpu_util_pct"] - random.uniform(0.5, 1.5) * ramp + random.gauss(0, 2), 5, 98))
        state["power_draw_w"] = float(np.clip(state["power_draw_w"] + random.uniform(1, 3) * ramp + random.gauss(0, 3), 80, 400))
        state["fan_speed_pct"] = float(np.clip(state["fan_speed_pct"] + random.uniform(0.8, 2) * ramp + random.gauss(0, 1), 20, 100))
        if ramp > 0.4: state["ecc_sbe_errors"] += random.randint(0, 3)
        if ramp > 0.7: state["ecc_dbe_errors"] += random.randint(0, 1)
        raw = {"gpu_temp_c": state["gpu_temp_c"], "gpu_util_pct": state["gpu_util_pct"], "gpu_mem_util_pct": float(np.clip(state["gpu_util_pct"] * .85 + random.gauss(0, 3), 0, 100)), "power_draw_w": state["power_draw_w"], "fan_speed_pct": state["fan_speed_pct"], "ecc_sbe_errors": state["ecc_sbe_errors"], "ecc_dbe_errors": state["ecc_dbe_errors"], "cpu_temp_c": state["gpu_temp_c"] * .7, "cpu_util_pct": state["gpu_util_pct"] * .6, "disk_io_mbps": random.uniform(40, 160), "network_io_mbps": random.uniform(80, 300), "pcie_replay_errors": state["ecc_sbe_errors"] // 4}
        sequence = np.tile(np.asarray([raw[name] for name in RAW_FEATURES], dtype=np.float32), (WINDOW, 1))
        sequences.append(sequence + np.random.normal(0, .3, sequence.shape).astype(np.float32))
        ids.append(device_id)
        raw_readings.append(raw)
    sequence_batch = np.stack(sequences)
    xgb, bigru, shap_values, _ = _score_batch(_models(request), sequence_batch, np.stack([_tabular(item) for item in sequence_batch]), explain=True)
    devices = []
    for device_id, raw, xgb_probability, bigru_probability, shap_row in zip(ids, raw_readings, xgb, bigru, shap_values):
        result = _format_prediction(xgb_probability, bigru_probability, shap_row, None)
        devices.append({"device_id": device_id, **result, "readings": {key: round(value, 2) for key, value in raw.items()}})
        if result["risk_level"] != "low":
            background_tasks.add_task(alerts.maybe_alert, device_id, result["risk_level"], result["ensemble_probability"], result["top_shap_reasons"], result["recommended_action"])
    devices.sort(key=lambda item: -item["ensemble_probability"])
    return {"timestamp": time.time(), "total_devices": len(devices), "high_risk": sum(item["risk_level"] == "high" for item in devices), "medium_risk": sum(item["risk_level"] == "medium" for item in devices), "devices": devices, "source": "live_simulator"}


@app.post("/actions/cordon_drain/{device_id}")
def action_cordon_drain(device_id: int, x_action_token: Annotated[str | None, Header()] = None):
    real_actions = os.environ.get("ENABLE_REAL_K8S_ACTIONS", "false").lower() == "true"
    configured_token = os.environ.get("ACTION_API_TOKEN", "")
    if real_actions:
        if not configured_token:
            raise HTTPException(status_code=503, detail="Real Kubernetes actions require ACTION_API_TOKEN")
        if not x_action_token or not hmac.compare_digest(x_action_token, configured_token):
            raise HTTPException(status_code=403, detail="Invalid action token")
    log = k8s_action.cordon_and_drain(device_id, real=real_actions)
    return {"device_id": device_id, "mode": log.mode, "log": log.lines}
