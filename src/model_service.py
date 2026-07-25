"""Process-safe hybrid model service used by the FastAPI application."""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import uuid
from pathlib import Path

import joblib
import numpy as np
import torch

from train_bigru_attention import BiGRUAttention
from xgb_worker import worker_main


class ModelServiceError(RuntimeError):
    """A model artifact or its isolated inference worker is unavailable."""


class XGBShapClient:
    """A serial RPC client for a child process that owns XGBoost and SHAP."""

    def __init__(self, model_path: Path, timeout_seconds: float = 30.0):
        self._context = mp.get_context("spawn")
        self._requests = self._context.Queue(maxsize=8)
        self._responses = self._context.Queue(maxsize=8)
        self._process = self._context.Process(
            target=worker_main,
            args=(self._requests, self._responses, str(model_path)),
            name="clusterhealth-xgb-shap",
            daemon=True,
        )
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()

    def start(self) -> None:
        self._process.start()
        try:
            message, detail = self._responses.get(timeout=self._timeout_seconds)
        except queue.Empty as exc:
            self.close()
            raise ModelServiceError("Timed out starting the XGBoost/SHAP worker") from exc
        if message != "ready":
            self.close()
            raise ModelServiceError(f"XGBoost/SHAP worker failed to start:\n{detail}")

    def predict(self, tab_rows: np.ndarray, *, explain: bool) -> tuple[np.ndarray, np.ndarray | None]:
        if not self._process.is_alive():
            raise ModelServiceError("The XGBoost/SHAP worker is not running")
        request_id = uuid.uuid4().hex
        with self._lock:
            self._requests.put((request_id, np.asarray(tab_rows, dtype=np.float32), explain))
            try:
                response = self._responses.get(timeout=self._timeout_seconds)
            except queue.Empty as exc:
                raise ModelServiceError("Timed out waiting for XGBoost/SHAP inference") from exc

        kind, returned_id, *payload = response
        if returned_id != request_id:
            raise ModelServiceError("Received an out-of-order XGBoost/SHAP worker response")
        if kind == "error":
            raise ModelServiceError(f"XGBoost/SHAP inference failed:\n{payload[0]}")
        if kind != "result":
            raise ModelServiceError(f"Unexpected XGBoost/SHAP worker response: {kind}")
        return payload[0], payload[1]

    def close(self) -> None:
        if self._process.is_alive():
            self._requests.put(None)
            self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        self._requests.close()
        self._responses.close()

    @property
    def is_healthy(self) -> bool:
        return self._process.is_alive()


class HybridModelService:
    """Owns CPU BiGRU inference and delegates XGBoost/SHAP to its worker."""

    def __init__(self, models_dir: Path):
        required = [
            models_dir / "xgboost_model.joblib",
            models_dir / "bigru_scaler.joblib",
            models_dir / "bigru_config.joblib",
            models_dir / "bigru_attention.pt",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ModelServiceError(f"Missing required model artifacts: {', '.join(missing)}")

        self.xgb = XGBShapClient(models_dir / "xgboost_model.joblib")
        self.xgb.start()
        try:
            self.scaler = joblib.load(models_dir / "bigru_scaler.joblib")
            config = joblib.load(models_dir / "bigru_config.joblib")
            self.n_features = int(config["n_features"])
            self.bigru = BiGRUAttention(
                n_features=self.n_features,
                hidden_dim=int(config["hidden_dim"]),
            )
            state_dict = torch.load(
                models_dir / "bigru_attention.pt", map_location="cpu", weights_only=True
            )
            self.bigru.load_state_dict(state_dict)
            self.bigru.eval()
            self._torch_lock = threading.Lock()
        except Exception:
            self.xgb.close()
            raise

    def score(self, seq_rows: np.ndarray, tab_rows: np.ndarray, *, explain: bool):
        sequences = np.asarray(seq_rows, dtype=np.float32)
        tabular = np.asarray(tab_rows, dtype=np.float32)
        if sequences.ndim != 3 or sequences.shape[-1] != self.n_features:
            raise ValueError(f"Expected sequences shaped (n, window, {self.n_features}), got {sequences.shape}")
        if tabular.ndim != 2 or tabular.shape[0] != sequences.shape[0]:
            raise ValueError("Tabular and sequential batches must have the same number of rows")
        if not np.isfinite(sequences).all() or not np.isfinite(tabular).all():
            raise ValueError("Model inputs must contain only finite numbers")

        xgb_probabilities, shap_values = self.xgb.predict(tabular, explain=explain)
        scaled = self.scaler.transform(sequences.reshape(-1, self.n_features))
        scaled = scaled.reshape(sequences.shape).astype(np.float32, copy=False)
        with self._torch_lock, torch.inference_mode():
            logits, attention = self.bigru(torch.from_numpy(scaled))
            bigru_probabilities = torch.sigmoid(logits).cpu().numpy()
            attention_weights = attention.cpu().numpy()
        return xgb_probabilities, bigru_probabilities, shap_values, attention_weights

    def close(self) -> None:
        self.xgb.close()
