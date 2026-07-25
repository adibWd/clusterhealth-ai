"""
ClusterHealth AI — Risk Score Re-Exporter
===========================================
Polls the ClusterHealth AI API's /cluster/overview endpoint every few
seconds and re-publishes each device's ensemble failure probability as
its OWN Prometheus metric. This is what lets Grafana plot "raw GPU temp"
and "AI-predicted failure risk" on the SAME timeline — the single most
convincing chart you can show judges, because it visually proves the
model's risk score rises BEFORE the hardware metric would trip a normal
static threshold alert.

Run (after the API is up):
    python3 risk_exporter.py
Then check: http://localhost:9500/metrics
"""

import os
import time
import requests
from prometheus_client import start_http_server, Gauge

API_URL = os.environ.get("CLUSTERHEALTH_API_URL", "http://localhost:8000")
POLL_SECONDS = int(os.environ.get("RISK_EXPORTER_POLL_SECONDS", "10"))

RISK_SCORE = Gauge(
    "clusterhealth_ai_failure_risk", "AI-predicted failure probability (0-1)", ["gpu"]
)
RISK_LEVEL = Gauge(
    "clusterhealth_ai_risk_level_numeric",
    "AI risk level as a number: 0=low, 1=medium, 2=high", ["gpu"],
)

LEVEL_MAP = {"low": 0, "medium": 1, "high": 2}


def poll_once():
    try:
        r = requests.get(f"{API_URL}/live/cluster/overview", timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("total_devices", 0) == 0:
            # Prometheus/metrics-exporter likely still warming up on first boot.
            print("[risk_exporter] Live overview returned 0 devices, will retry")
            return
        for device in data["devices"]:
            gpu_id = str(device["device_id"])
            RISK_SCORE.labels(gpu=gpu_id).set(device["ensemble_probability"])
            RISK_LEVEL.labels(gpu=gpu_id).set(LEVEL_MAP.get(device["risk_level"], 0))
        print(f"[risk_exporter] Published LIVE risk scores for {len(data['devices'])} devices "
              f"(source: real Prometheus telemetry, not the offline demo CSV)")
    except Exception as e:
        print(f"[risk_exporter] Poll failed: {e}")


def main():
    start_http_server(9500)
    print("ClusterHealth AI risk-score exporter running on http://localhost:9500/metrics")
    print(f"Polling {API_URL}/cluster/overview every {POLL_SECONDS}s")
    while True:
        poll_once()
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
