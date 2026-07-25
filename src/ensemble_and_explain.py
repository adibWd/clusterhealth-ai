"""Build the XGBoost/BiGRU ensemble and its SHAP explanation artifact.

On macOS, the XGBoost and PyTorch wheels can load incompatible OpenMP runtimes
into one interpreter.  XGBoost + SHAP therefore run in a short-lived child
process; PyTorch runs only in the parent process. Explainability metadata is
written as JSON, so the API never unpickles an XGBoost object in its Torch
process.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


BASE = Path(__file__).resolve().parent.parent
W_XGB, W_BIGRU = 0.6, 0.4


def _xgb_shap_worker(input_path: Path, output_path: Path, artifact_path: Path) -> None:
    """Load XGBoost and SHAP in a process that never imports PyTorch."""
    import shap

    from features import tabular_feature_names

    payload = np.load(input_path)
    tab_x = payload["tab_x"]
    sample_idx = payload["sample_idx"]

    # Importing/unpickling this model is deliberately confined to this process.
    xgb_model = joblib.load(BASE / "models" / "xgboost_model.joblib")
    xgb_proba_all = xgb_model.predict_proba(tab_x)[:, 1]

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = np.asarray(explainer.shap_values(tab_x[sample_idx]))
    # SHAP currently returns (samples, features) for this binary classifier.
    # Retain the positive-class values if a future SHAP release returns a class axis.
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 1]

    feature_names = tabular_feature_names()
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = sorted(zip(feature_names, mean_abs_shap), key=lambda item: -item[1])[:10]

    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "feature_names": feature_names,
                "top_features": [{"name": name, "mean_abs_shap": float(value)} for name, value in top_features],
                "weights": {"xgb": W_XGB, "bigru": W_BIGRU},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_path,
        xgb_proba_all=xgb_proba_all,
        top_feature_names=np.asarray([name for name, _ in top_features]),
        top_feature_values=np.asarray([value for _, value in top_features]),
    )


def _run_xgb_and_shap(tab_x: np.ndarray, sample_idx: np.ndarray, models_dir: Path):
    """Return XGBoost probabilities and SHAP ranking without loading it beside Torch."""
    with tempfile.TemporaryDirectory(prefix="clusterhealth-xgb-") as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "input.npz"
        output_path = temp_dir / "output.npz"
        np.savez_compressed(input_path, tab_x=tab_x, sample_idx=sample_idx)

        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--xgb-shap-worker",
            str(input_path),
            str(output_path),
            str(models_dir / "ensemble_explain.json"),
        ]
        try:
            subprocess.run(command, cwd=BASE, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "The isolated XGBoost/SHAP worker failed.\n"
                f"stdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
            ) from exc

        result = np.load(output_path)
        top_features = list(zip(result["top_feature_names"].tolist(), result["top_feature_values"].tolist()))
        return result["xgb_proba_all"], top_features


def main() -> None:
    # These imports are intentionally delayed so --xgb-shap-worker never loads Torch.
    import torch

    from features import build_windows
    from train_bigru_attention import BiGRUAttention

    df = pd.read_csv(BASE / "data" / "telemetry.csv")
    seq_x, tab_x, y, meta = build_windows(df)
    devices = meta["device_id"].unique()
    _, test_dev = train_test_split(devices, test_size=0.25, random_state=42)
    test_mask = meta["device_id"].isin(test_dev).to_numpy()
    models_dir = BASE / "models"

    # XGBoost and SHAP stay out of this interpreter, avoiding the libomp collision.
    sample_idx = np.flatnonzero(test_mask)[:500]
    xgb_proba_all, top_features = _run_xgb_and_shap(tab_x, sample_idx, models_dir)

    scaler = joblib.load(models_dir / "bigru_scaler.joblib")
    cfg = joblib.load(models_dir / "bigru_config.joblib")
    bigru = BiGRUAttention(n_features=cfg["n_features"], hidden_dim=cfg["hidden_dim"])
    bigru.load_state_dict(torch.load(models_dir / "bigru_attention.pt", map_location="cpu", weights_only=True))
    bigru.eval()

    shape = seq_x.shape
    seq_scaled = scaler.transform(seq_x.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
    with torch.no_grad():
        logits, _ = bigru(torch.from_numpy(seq_scaled))
        bigru_proba_all = torch.sigmoid(logits).cpu().numpy()

    ensemble_proba_all = W_XGB * xgb_proba_all + W_BIGRU * bigru_proba_all
    y_test = y[test_mask]
    ens_test = ensemble_proba_all[test_mask]
    preds = (ens_test >= 0.5).astype(int)

    print("=== Weighted Ensemble (0.6*XGB + 0.4*BiGRU) ===")
    print(f"ROC-AUC : {roc_auc_score(y_test, ens_test):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_test, ens_test):.4f}")
    print(classification_report(y_test, preds, digits=3))
    print("\nTop 10 features driving failure predictions (mean |SHAP value|):")
    for name, value in top_features:
        print(f"  {name:30s} {value:.4f}")
    print(f"\nSaved ensemble/explainability metadata to {models_dir / 'ensemble_explain.json'}")


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--xgb-shap-worker":
        _xgb_shap_worker(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        main()
