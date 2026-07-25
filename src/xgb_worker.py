"""XGBoost/SHAP worker kept deliberately separate from the Torch process.

The Apple-Silicon wheels used by this project load incompatible OpenMP
runtimes when PyTorch and XGBoost share an interpreter.  Do not import Torch
in this module.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def _positive_class_shap(values: Any) -> np.ndarray:
    """Normalise SHAP's version-dependent binary-class output to 2-D."""
    array = np.asarray(values)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] == 2:
        return array[..., 1]
    if array.ndim == 3 and array.shape[0] == 2:
        return array[1]
    raise ValueError(f"Unexpected SHAP output shape: {array.shape}")


def worker_main(request_queue: Any, response_queue: Any, model_path: str) -> None:
    """Serve serial prediction/explanation requests in an XGBoost-only process."""
    try:
        import shap

        # joblib/pickle artifacts are executable code.  Only load model files
        # produced and protected by this deployment.
        model = joblib.load(Path(model_path))
        explainer = shap.TreeExplainer(model)
        response_queue.put(("ready", None))
    except Exception:
        response_queue.put(("startup_error", traceback.format_exc()))
        return

    while True:
        request = request_queue.get()
        if request is None:
            return

        request_id, tab_rows, explain = request
        try:
            rows = np.asarray(tab_rows, dtype=np.float32)
            if rows.ndim != 2:
                raise ValueError(f"Expected a 2-D tabular matrix, got {rows.shape}")
            probabilities = model.predict_proba(rows)[:, 1].astype(np.float32, copy=False)
            shap_values = _positive_class_shap(explainer.shap_values(rows)) if explain else None
            response_queue.put(("result", request_id, probabilities, shap_values))
        except Exception:
            response_queue.put(("error", request_id, traceback.format_exc()))
