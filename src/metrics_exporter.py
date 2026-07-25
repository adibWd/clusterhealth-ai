"""
ClusterHealth AI — Live Metrics Exporter
=========================================
Exposes GPU/CPU telemetry as real Prometheus metrics on :9400/metrics,
using the SAME metric names a real NVIDIA DCGM Exporter + Node Exporter
would use. Prometheus scrapes this; Grafana visualizes it; the
ClusterHealth AI API queries it for live predictions.

Why this exists: you don't have a physical GPU cluster for the hackathon,
but Prometheus doesn't care where metrics come from — it just scrapes an
HTTP endpoint. This script *is* that endpoint, continuously simulating
300 devices evolving in real time (including live failure events), so
your entire monitoring stack (Prometheus -> Grafana -> ClusterHealth AI)
runs exactly as it would against a real cluster.

Run:
    python3 metrics_exporter.py
Then check: http://localhost:9400/metrics
"""

import time
import random
import threading
from prometheus_client import start_http_server, Gauge

N_DEVICES = 30           # fewer than the offline dataset -- kept light for a live demo
TICK_SECONDS = 5         # how often metrics update (real DCGM scrapes every 10-30s)
FAILURE_CHANCE_PER_TICK = 0.004   # chance any healthy device starts degrading each tick

# ---- Prometheus gauges, named exactly like real DCGM / Node Exporter metrics ----
GPU_TEMP = Gauge("DCGM_FI_DEV_GPU_TEMP", "GPU temperature (C)", ["gpu"])
GPU_UTIL = Gauge("DCGM_FI_DEV_GPU_UTIL", "GPU utilization (%)", ["gpu"])
GPU_MEM_UTIL = Gauge("DCGM_FI_DEV_MEM_COPY_UTIL", "GPU memory utilization (%)", ["gpu"])
POWER_USAGE = Gauge("DCGM_FI_DEV_POWER_USAGE", "GPU power draw (W)", ["gpu"])
ECC_SBE = Gauge("DCGM_FI_DEV_ECC_SBE_VOL_TOTAL", "ECC single-bit errors (total)", ["gpu"])
ECC_DBE = Gauge("DCGM_FI_DEV_ECC_DBE_VOL_TOTAL", "ECC double-bit errors (total)", ["gpu"])
PCIE_REPLAY = Gauge("DCGM_FI_DEV_PCIE_REPLAY_COUNTER", "PCIe replay errors (total)", ["gpu"])
FAN_SPEED = Gauge("node_hwmon_fan_speed_percent", "Fan speed (%)", ["gpu"])
CPU_TEMP = Gauge("node_hwmon_temp_celsius", "CPU temperature (C)", ["gpu"])
CPU_UTIL = Gauge("node_cpu_util_percent", "CPU utilization (%)", ["gpu"])
DISK_IO = Gauge("node_disk_io_mbps", "Disk I/O (MB/s)", ["gpu"])
NET_IO = Gauge("node_network_io_mbps", "Network I/O (MB/s)", ["gpu"])

# ---- also expose the AI's own risk score once ClusterHealth API scores it ----
# (populated by risk_exporter.py -- see that file -- kept separate so this
#  exporter only ever represents "raw hardware telemetry", exactly like a
#  real DCGM/Node Exporter would; the AI layer publishes its own metric)


class DeviceState:
    def __init__(self, gpu_id: str):
        self.gpu_id = gpu_id
        self.reset_healthy()
        self.degrading = False
        self.degrade_ticks = 0

    def reset_healthy(self):
        self.gpu_temp = random.gauss(62, 3)
        self.gpu_util = min(100, max(0, random.gauss(70, 15)))
        self.gpu_mem_util = min(100, max(0, random.gauss(55, 12)))
        self.power = random.gauss(220, 20)
        self.fan = min(100, max(0, random.gauss(45, 8)))
        self.ecc_sbe = 0.0
        self.ecc_dbe = 0.0
        self.pcie_replay = 0.0
        self.cpu_temp = random.gauss(48, 4)
        self.cpu_util = min(100, max(0, random.gauss(40, 15)))
        self.disk_io = max(0, random.gauss(80, 25))
        self.net_io = max(0, random.gauss(150, 40))

    def tick(self):
        if not self.degrading:
            # small healthy jitter
            self.gpu_temp += random.gauss(0, 0.6)
            self.gpu_util = min(100, max(0, self.gpu_util + random.gauss(0, 3)))
            self.power += random.gauss(0, 3)
            self.fan += random.gauss(0, 1.5)
            self.cpu_util = min(100, max(0, self.cpu_util + random.gauss(0, 3)))
            self.disk_io = max(0, self.disk_io + random.gauss(0, 5))
            self.net_io = max(0, self.net_io + random.gauss(0, 8))

            if random.random() < FAILURE_CHANCE_PER_TICK:
                self.degrading = True
                self.degrade_ticks = 0
        else:
            # active degradation: thermal creep + ECC error burst + throttling
            self.degrade_ticks += 1
            self.gpu_temp += random.uniform(0.4, 1.2)
            self.fan += random.uniform(0.8, 2.0)
            self.power += random.uniform(1.0, 3.0)
            self.ecc_sbe += random.choice([0, 0, 1, 1, 2])
            if self.degrade_ticks > 8:
                self.ecc_dbe += random.choice([0, 0, 1])
                self.gpu_util = max(5, self.gpu_util * 0.85)
            self.pcie_replay += random.choice([0, 0, 1])

            # after enough ticks, device "fails" -> gets replaced/recovered
            if self.degrade_ticks > 20:
                self.reset_healthy()
                self.degrading = False

    def publish(self):
        GPU_TEMP.labels(gpu=self.gpu_id).set(self.gpu_temp)
        GPU_UTIL.labels(gpu=self.gpu_id).set(self.gpu_util)
        GPU_MEM_UTIL.labels(gpu=self.gpu_id).set(self.gpu_mem_util)
        POWER_USAGE.labels(gpu=self.gpu_id).set(self.power)
        ECC_SBE.labels(gpu=self.gpu_id).set(self.ecc_sbe)
        ECC_DBE.labels(gpu=self.gpu_id).set(self.ecc_dbe)
        PCIE_REPLAY.labels(gpu=self.gpu_id).set(self.pcie_replay)
        FAN_SPEED.labels(gpu=self.gpu_id).set(self.fan)
        CPU_TEMP.labels(gpu=self.gpu_id).set(self.cpu_temp)
        CPU_UTIL.labels(gpu=self.gpu_id).set(self.cpu_util)
        DISK_IO.labels(gpu=self.gpu_id).set(self.disk_io)
        NET_IO.labels(gpu=self.gpu_id).set(self.net_io)


def simulation_loop(devices):
    while True:
        for d in devices:
            d.tick()
            d.publish()
        time.sleep(TICK_SECONDS)


def main():
    devices = [DeviceState(str(i)) for i in range(N_DEVICES)]
    for d in devices:
        d.publish()

    start_http_server(9400)
    print(f"ClusterHealth AI metrics exporter running on http://localhost:9400/metrics")
    print(f"Simulating {N_DEVICES} devices, updating every {TICK_SECONDS}s")

    t = threading.Thread(target=simulation_loop, args=(devices,), daemon=True)
    t.start()
    t.join()


if __name__ == "__main__":
    main()
