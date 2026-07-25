"""
ClusterHealth AI — Live Prometheus Data Fetcher
=================================================
Queries a REAL running Prometheus instance for GPU/CPU telemetry and
assembles the (WINDOW, n_features) array the models expect — this is
the bridge between "synthetic CSV demo" and "actually monitoring a live
cluster" (or, for the hackathon, our own metrics_exporter.py standing in
for a real DCGM/Node Exporter).

Metric-name mapping (matches real DCGM Exporter / Node Exporter names):
    gpu_temp_c          <- DCGM_FI_DEV_GPU_TEMP
    gpu_util_pct        <- DCGM_FI_DEV_GPU_UTIL
    gpu_mem_util_pct    <- DCGM_FI_DEV_MEM_COPY_UTIL
    power_draw_w        <- DCGM_FI_DEV_POWER_USAGE
    fan_speed_pct       <- node_hwmon_fan_speed_percent
    ecc_sbe_errors      <- DCGM_FI_DEV_ECC_SBE_VOL_TOTAL
    ecc_dbe_errors      <- DCGM_FI_DEV_ECC_DBE_VOL_TOTAL
    cpu_temp_c          <- node_hwmon_temp_celsius
    cpu_util_pct        <- node_cpu_util_percent
    disk_io_mbps        <- node_disk_io_mbps
    network_io_mbps     <- node_network_io_mbps
    pcie_replay_errors  <- DCGM_FI_DEV_PCIE_REPLAY_COUNTER
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import requests

from features import RAW_FEATURES, WINDOW

PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")


class LiveDataUnavailable(RuntimeError):
    """Prometheus could not provide a complete, trustworthy model window."""

METRIC_MAP = {
    "gpu_temp_c": "DCGM_FI_DEV_GPU_TEMP",
    "gpu_util_pct": "DCGM_FI_DEV_GPU_UTIL",
    "gpu_mem_util_pct": "DCGM_FI_DEV_MEM_COPY_UTIL",
    "power_draw_w": "DCGM_FI_DEV_POWER_USAGE",
    "fan_speed_pct": "node_hwmon_fan_speed_percent",
    "ecc_sbe_errors": "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL",
    "ecc_dbe_errors": "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL",
    "cpu_temp_c": "node_hwmon_temp_celsius",
    "cpu_util_pct": "node_cpu_util_percent",
    "disk_io_mbps": "node_disk_io_mbps",
    "network_io_mbps": "node_network_io_mbps",
    "pcie_replay_errors": "DCGM_FI_DEV_PCIE_REPLAY_COUNTER",
}


def prometheus_up() -> bool:
    try:
        r = requests.get(f"{PROMETHEUS_URL}/-/healthy", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def list_live_devices() -> list[str]:
    """Ask Prometheus which device (gpu label) IDs currently have data."""
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": "DCGM_FI_DEV_GPU_TEMP"},
            timeout=5,
        )
        r.raise_for_status()
        result = r.json()["data"]["result"]
        devices = {item.get("metric", {}).get("gpu") for item in result}
        return sorted((device for device in devices if device), key=lambda value: (not value.isdigit(), value))
    except Exception as e:
        print(f"[live_data] Could not list devices from Prometheus: {e}")
        return []


def _query_range(promql: str, minutes: int, step_seconds: int):
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    r = requests.get(
        f"{PROMETHEUS_URL}/api/v1/query_range",
        params={
            "query": promql,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": f"{step_seconds}s",
        },
        timeout=10,
    )
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "success":
        raise LiveDataUnavailable("Prometheus returned an unsuccessful query response")
    result = payload.get("data", {}).get("result", [])
    if not result:
        return None
    values = result[0]["values"]
    series = np.asarray([float(value) for _, value in values], dtype=np.float32)
    if not np.isfinite(series).all():
        raise LiveDataUnavailable("Prometheus returned non-finite telemetry")
    return series


def build_live_window(device_id: str, step_seconds: int = 5) -> np.ndarray:
    """
    Fetch the last WINDOW timesteps for every feature from Prometheus for
    one device and assemble the (WINDOW, n_features) array the models expect.
    Missing telemetry is an error: silently substituting zeros can turn an
    upstream outage into a dangerous false prediction.
    """
    minutes = max(1, (WINDOW * step_seconds) // 60 + 1)
    escaped_device_id = device_id.replace("\\", "\\\\").replace('"', '\\"')

    def fetch(feature: str) -> tuple[str, np.ndarray | None]:
        promql = f'{METRIC_MAP[feature]}{{gpu="{escaped_device_id}"}}'
        return feature, _query_range(promql, minutes=minutes, step_seconds=step_seconds)

    series_by_feature: dict[str, np.ndarray | None] = {}
    try:
        with ThreadPoolExecutor(max_workers=min(6, len(RAW_FEATURES))) as executor:
            futures = [executor.submit(fetch, feature) for feature in RAW_FEATURES]
            for future in as_completed(futures):
                feature, series = future.result()
                series_by_feature[feature] = series
    except (requests.RequestException, ValueError, KeyError) as exc:
        raise LiveDataUnavailable(f"Could not query Prometheus: {exc}") from exc

    missing = [feature for feature in RAW_FEATURES if series_by_feature.get(feature) is None]
    if missing:
        raise LiveDataUnavailable(f"Missing live telemetry for: {', '.join(missing)}")

    columns = []
    for feature in RAW_FEATURES:
        series = series_by_feature[feature]
        assert series is not None
        if len(series) < WINDOW:
            series = np.pad(series, (WINDOW - len(series), 0), mode="edge")
        columns.append(series[-WINDOW:])
    return np.stack(columns, axis=1).astype(np.float32)  # (WINDOW, n_features)
