"""
ClusterHealth AI - Synthetic Telemetry Generator
=================================================
Real Prometheus / NVIDIA DCGM / Node Exporter data isn't available for a
hackathon demo, so this script generates REALISTIC synthetic GPU/CPU cluster
telemetry with injected pre-failure degradation patterns.

Why synthetic data is a legitimate hackathon strategy:
  - Judges care about the AI pipeline, explainability, and system design,
    not whether you have a physical GPU farm.
  - The generator encodes real failure physics (thermal creep, ECC error
    bursts, throttling, memory leaks) so the model learns genuine signal.
  - In your pitch, say explicitly: "trained on synthetic telemetry modeled
    after real DCGM failure signatures; the same pipeline plugs directly
    into live Prometheus/DCGM data with zero code changes to the model."

Output: data/telemetry.csv (long format: one row per device per timestep)
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

N_DEVICES = 300          # number of GPUs/nodes being monitored
T = 96                   # timesteps per device (96 * 15min = 24h of history)
FAILURE_RATE = 0.18      # fraction of devices that experience a failure event
HORIZON = 8              # predict failure within next 8 timesteps (~2h)

FEATURES = [
    "gpu_temp_c", "gpu_util_pct", "gpu_mem_util_pct", "power_draw_w",
    "fan_speed_pct", "ecc_sbe_errors", "ecc_dbe_errors", "cpu_temp_c",
    "cpu_util_pct", "disk_io_mbps", "network_io_mbps", "pcie_replay_errors",
]


def simulate_device(device_id: int, will_fail: bool):
    """Simulate one device's telemetry for T timesteps."""
    t = np.arange(T)

    # --- baseline healthy operating ranges ---
    gpu_temp = RNG.normal(62, 3, T)
    gpu_util = np.clip(RNG.normal(70, 15, T), 0, 100)
    gpu_mem = np.clip(RNG.normal(55, 12, T), 0, 100)
    power = RNG.normal(220, 20, T)
    fan = np.clip(RNG.normal(45, 8, T), 0, 100)
    ecc_sbe = RNG.poisson(0.2, T).astype(float)
    ecc_dbe = np.zeros(T)
    cpu_temp = RNG.normal(48, 4, T)
    cpu_util = np.clip(RNG.normal(40, 15, T), 0, 100)
    disk_io = RNG.normal(80, 25, T)
    net_io = RNG.normal(150, 40, T)
    pcie_err = RNG.poisson(0.05, T).astype(float)

    label = np.zeros(T, dtype=int)
    failure_step = None

    if will_fail:
        # Failure occurs close to "now" (end of series) so that a live
        # monitoring snapshot (the most recent window) actually shows the
        # device mid-degradation -- this is what the demo dashboard scores.
        failure_step = RNG.integers(T - 12, T - 1)
        onset = max(0, failure_step - 20)  # degradation begins ~20 steps before

        # thermal creep + throttling
        ramp = np.clip((t - onset) / max(1, (failure_step - onset)), 0, 1)
        gpu_temp += ramp * RNG.uniform(15, 25) * (t >= onset)
        fan += ramp * RNG.uniform(20, 35) * (t >= onset)
        power += ramp * RNG.uniform(30, 60) * (t >= onset)

        # ECC error burst (classic pre-failure signature)
        burst_mask = (t >= onset) & (t < failure_step)
        ecc_sbe[burst_mask] += RNG.poisson(3, burst_mask.sum())
        ecc_dbe[(t >= failure_step - 5) & (t < failure_step)] += RNG.poisson(1, 5)
        pcie_err[burst_mask] += RNG.poisson(0.5, burst_mask.sum())

        # utilization drops as the card throttles / workloads get evicted
        gpu_util[(t >= failure_step - 8) & (t < failure_step)] *= 0.4

        # label: 1 from HORIZON steps before the failure onward (covers both
        # the "about to fail" warning window AND the "currently degraded /
        # already failed" state -- a live monitoring system needs to flag
        # both as requiring action, not just the pre-failure moment).
        label[t >= failure_step - HORIZON] = 1

    df = pd.DataFrame({
        "device_id": device_id,
        "timestep": t,
        "gpu_temp_c": gpu_temp,
        "gpu_util_pct": np.clip(gpu_util, 0, 100),
        "gpu_mem_util_pct": np.clip(gpu_mem, 0, 100),
        "power_draw_w": np.clip(power, 50, 400),
        "fan_speed_pct": np.clip(fan, 0, 100),
        "ecc_sbe_errors": np.clip(ecc_sbe, 0, None),
        "ecc_dbe_errors": np.clip(ecc_dbe, 0, None),
        "cpu_temp_c": cpu_temp,
        "cpu_util_pct": np.clip(cpu_util, 0, 100),
        "disk_io_mbps": np.clip(disk_io, 0, None),
        "network_io_mbps": np.clip(net_io, 0, None),
        "pcie_replay_errors": np.clip(pcie_err, 0, None),
        "failure_within_horizon": label,
        "will_fail_device": int(will_fail),
    })
    return df


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    n_fail = int(N_DEVICES * FAILURE_RATE)
    fail_flags = np.array([True] * n_fail + [False] * (N_DEVICES - n_fail))
    RNG.shuffle(fail_flags)

    frames = [simulate_device(i, bool(fail_flags[i])) for i in range(N_DEVICES)]
    full = pd.concat(frames, ignore_index=True)

    out_path = out_dir / "telemetry.csv"
    full.to_csv(out_path, index=False)

    print(f"Generated {len(full):,} rows across {N_DEVICES} devices")
    print(f"Devices with a failure event: {n_fail} ({FAILURE_RATE:.0%})")
    print(f"Positive label rate (failure_within_horizon=1): "
          f"{full['failure_within_horizon'].mean():.3%}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
