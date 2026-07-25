"""
Report Card & Threshold Tuner
==============================
Final-round challenge module (AI Innovation Hackathon 2026, DIU CSE) --
"Team NephroAI: ClusterHealth AI, Predictive and Explainable GPU Cluster
Management".

Problem this closes: the alert level in ClusterHealth AI was fixed by the
team, not the operator. A strict level misses real faults; a loose level
floods the screen with false alarms; nobody could see that trade-off before
choosing. This module makes the trade-off visible and adjustable.

Design, in one pass:
  1. SCORE ONCE.  Every labeled window in the stored synthetic telemetry
     dataset (data/telemetry.csv) is scored exactly once, by the SAME
     already-trained XGBoost + BiGRU-Attention ensemble the rest of the
     API uses (see api.py / model_service.py). This is inference only --
     no model is trained or fine-tuned here or anywhere in this module,
     satisfying the brief's "no long model training inside the session".
  2. CACHE IT.  The resulting (probability, ground_truth) arrays are cached
     in-process (see api.py's lifespan hook), so moving the threshold
     slider is a ~microsecond NumPy comparison, not a re-inference call --
     this is what makes the live "Threshold Tuner" demo feel instant.
  3. REPORT CARD.  For any alert threshold, count three operator-facing
     numbers against the dataset's injected ground-truth failure labels:
       - faults_caught  = true positives  (real fault, alert fired)
       - faults_missed  = false negatives (real fault, no alert)
       - false_alarms   = false positives (alert fired, no real fault)
  4. SUGGESTED LEVEL.  Sweep the same cached data across a threshold grid
     and recommend the single threshold that maximizes an F-beta score
     with beta > 1 -- i.e. it weighs catching real faults higher than
     avoiding false alarms, because a missed GPU/CPU failure in production
     is materially more expensive (downtime, possible data loss, emergency
     response) than one extra investigation. The reason string is built
     directly from that threshold's own counts, so it is always numerically
     honest, never a canned sentence.

Everything here is additive: a new APIRouter mounted onto the existing
FastAPI app (see api.py's `app.include_router(report_card.router)`). No
existing endpoint, model, or file is modified to make this work.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from features import WINDOW, build_windows
from model_service import HybridModelService, ModelServiceError

router = APIRouter(prefix="/reportcard", tags=["Report Card & Threshold Tuner"])

# Kept identical to api.py's ensemble weights on purpose (see api.py::W_XGB /
# W_BIGRU) -- the Report Card must score with the exact same ensemble the
# rest of the product uses, or its counts would silently drift from what the
# operator sees on the main dashboard.
W_XGB, W_BIGRU = 0.6, 0.4

MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.95
DEFAULT_THRESHOLD = 0.40
GRID_STEP = 0.01

# Convenience presets surfaced as one-click buttons in the dashboard tuner,
# in addition to the free-moving slider -- these are shortcuts, not the only
# alert levels an operator can choose.
LEVEL_PRESETS = {"low": 0.25, "medium": 0.50, "high": 0.75}

# A missed hardware fault (unplanned downtime, possible data loss, emergency
# on-call response) costs materially more than one extra false-alarm
# investigation. F-beta with beta > 1 encodes that asymmetry by weighing
# recall (fault coverage) higher than precision when scoring candidate
# alert levels. beta=2 -> recall counted ~4x as important as precision.
SUGGESTION_BETA = 2.0


class ReportCardUnavailable(RuntimeError):
    """The stored-run cache could not be built or is not ready yet."""


@dataclass(frozen=True)
class ReportCardCache:
    """Every stored, labeled window scored once by the hybrid ensemble."""

    probabilities: np.ndarray  # shape (n_windows,), ensemble P(failure)
    y_true: np.ndarray         # shape (n_windows,), ground-truth label
    total_windows: int
    total_labeled_faults: int


def build_report_card_cache(models: HybridModelService, telemetry: pd.DataFrame) -> ReportCardCache:
    """Score the full stored dataset once. Inference only -- see module docstring."""
    try:
        seq_x, tab_x, y, _meta = build_windows(telemetry, window=WINDOW)
        xgb_p, bigru_p, _shap, _attention = models.score(seq_x, tab_x, explain=False)
    except (ValueError, ModelServiceError) as exc:
        raise ReportCardUnavailable(f"Could not score the stored dataset: {exc}") from exc
    probabilities = (W_XGB * xgb_p + W_BIGRU * bigru_p).astype(np.float32)
    y_true = y.astype(np.int64)
    return ReportCardCache(
        probabilities=probabilities,
        y_true=y_true,
        total_windows=int(len(y_true)),
        total_labeled_faults=int(y_true.sum()),
    )


def _cache(request: Request) -> ReportCardCache:
    cache = getattr(request.app.state, "report_card_cache", None)
    if cache is None:
        raise HTTPException(
            status_code=503,
            detail="Report Card is still warming up -- the stored run has not finished scoring yet.",
        )
    return cache


def _counts(cache: ReportCardCache, threshold: float) -> dict:
    predicted = cache.probabilities >= threshold
    actual = cache.y_true.astype(bool)

    faults_caught = int(np.sum(predicted & actual))
    faults_missed = int(np.sum(~predicted & actual))
    false_alarms = int(np.sum(predicted & ~actual))
    correctly_quiet = int(np.sum(~predicted & ~actual))

    total_faults = faults_caught + faults_missed
    total_healthy = false_alarms + correctly_quiet
    precision = faults_caught / (faults_caught + false_alarms) if (faults_caught + false_alarms) else 0.0
    recall = faults_caught / total_faults if total_faults else 0.0
    false_alarm_rate = false_alarms / total_healthy if total_healthy else 0.0

    return {
        "threshold": round(float(threshold), 4),
        "faults_caught": faults_caught,
        "faults_missed": faults_missed,
        "false_alarms": false_alarms,
        "correctly_quiet": correctly_quiet,
        "total_labeled_faults": total_faults,
        "total_windows": cache.total_windows,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_alarm_rate": round(false_alarm_rate, 4),
    }


def _fbeta(precision: float, recall: float, beta: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta * beta
    denominator = (beta_sq * precision) + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denominator


def _suggest(cache: ReportCardCache) -> dict:
    grid = np.round(np.arange(MIN_THRESHOLD, MAX_THRESHOLD + 1e-9, GRID_STEP), 4)
    scored_rows = []
    for candidate in grid:
        row = _counts(cache, float(candidate))
        row["_fbeta"] = _fbeta(row["precision"], row["recall"], SUGGESTION_BETA)
        scored_rows.append(row)

    best_score = max(row["_fbeta"] for row in scored_rows)
    # The saturated, near-binary probabilities called out in the README's
    # calibration caveat mean several thresholds can tie on F-beta. Among
    # ties, prefer fewer false alarms, then the level closest to the
    # midpoint of the sweep range -- a stable, explainable tie-break rather
    # than an arbitrary "first match".
    EPS = 1e-9
    tied = [row for row in scored_rows if row["_fbeta"] >= best_score - EPS]
    midpoint = (MIN_THRESHOLD + MAX_THRESHOLD) / 2
    tied.sort(key=lambda row: (row["false_alarms"], abs(row["threshold"] - midpoint)))
    winner = dict(tied[0])
    winner.pop("_fbeta")

    reason = (
        f"Level {winner['threshold']:.0%} is suggested: across the stored run it catches "
        f"{winner['faults_caught']} of {winner['total_labeled_faults']} known faults "
        f"({winner['recall']:.0%} recall) while limiting false alarms to "
        f"{winner['false_alarms']}, the best recall-weighted balance found by sweeping "
        f"{MIN_THRESHOLD:.0%}-{MAX_THRESHOLD:.0%} in {GRID_STEP:.0%} steps."
    )
    winner["reason"] = reason
    winner["beta"] = SUGGESTION_BETA
    return winner


@router.get("/config")
def report_card_config(request: Request):
    """Static bounds + presets the dashboard needs to draw the tuner control."""
    cache = _cache(request)
    return {
        "min_threshold": MIN_THRESHOLD,
        "max_threshold": MAX_THRESHOLD,
        "default_threshold": DEFAULT_THRESHOLD,
        "grid_step": GRID_STEP,
        "presets": LEVEL_PRESETS,
        "total_windows": cache.total_windows,
        "total_labeled_faults": cache.total_labeled_faults,
        "beta": SUGGESTION_BETA,
        "data_source": "stored_run",
    }


@router.get("")
def report_card(request: Request, threshold: float = DEFAULT_THRESHOLD):
    """Report Card: recompute the three counts at the given alert threshold."""
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=422, detail="threshold must be between 0.0 and 1.0")
    cache = _cache(request)
    return _counts(cache, threshold)


@router.get("/suggested")
def report_card_suggested(request: Request):
    """Suggested Level: one recommended threshold plus a one-sentence, numbers-backed reason."""
    cache = _cache(request)
    return _suggest(cache)
