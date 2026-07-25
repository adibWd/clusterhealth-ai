"""
Train the tabular XGBoost failure-prediction model on rolling-window
statistics of GPU/CPU telemetry.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import xgboost as xgb

from features import build_windows, tabular_feature_names

BASE = Path(__file__).resolve().parent.parent


def main():
    df = pd.read_csv(BASE / "data" / "telemetry.csv")
    seq_X, tab_X, y, meta = build_windows(df)

    # split by DEVICE (not by row) so we never leak the same device's
    # timesteps across train/test -- avoids optimistic leakage.
    devices = meta["device_id"].unique()
    train_dev, test_dev = train_test_split(devices, test_size=0.25, random_state=42)
    train_mask = meta["device_id"].isin(train_dev).values
    test_mask = ~train_mask

    X_train, X_test = tab_X[train_mask], tab_X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print("=== XGBoost Failure Prediction ===")
    print(f"ROC-AUC : {roc_auc_score(y_test, proba):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_test, proba):.4f}")
    print(classification_report(y_test, preds, digits=3))

    models_dir = BASE / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "xgboost_model.joblib")
    joblib.dump(tabular_feature_names(), models_dir / "xgb_feature_names.joblib")

    # save test split artifacts so ensemble.py can reuse the exact same split
    np.savez(
        models_dir / "xgb_split.npz",
        test_mask=test_mask, train_mask=train_mask,
    )
    print(f"Saved model to {models_dir / 'xgboost_model.joblib'}")


if __name__ == "__main__":
    main()
