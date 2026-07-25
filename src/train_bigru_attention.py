"""
Train the BiGRU-Attention sequential model on raw telemetry windows.

Architecture (matches the concept-note diagram):
  Input (window, n_features)
    -> Bidirectional GRU (captures temporal degradation patterns)
    -> Additive Attention (learns which timesteps matter most)
    -> Dense classifier -> failure probability
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

from features import build_windows, RAW_FEATURES

BASE = Path(__file__).resolve().parent.parent
torch.manual_seed(42)


class Attention(nn.Module):
    """Additive (Bahdanau-style) attention over GRU hidden states."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, gru_out):
        # gru_out: (batch, seq_len, hidden_dim)
        scores = self.attn(gru_out).squeeze(-1)        # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)          # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), gru_out).squeeze(1)  # (batch, hidden_dim)
        return context, weights


class BiGRUAttention(nn.Module):
    def __init__(self, n_features, hidden_dim=32, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
        )
        self.attention = Attention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)                # (batch, seq_len, hidden_dim*2)
        context, attn_weights = self.attention(gru_out)
        logits = self.classifier(context).squeeze(-1)
        return logits, attn_weights


def main():
    df = pd.read_csv(BASE / "data" / "telemetry.csv")
    seq_X, tab_X, y, meta = build_windows(df)

    devices = meta["device_id"].unique()
    train_dev, test_dev = train_test_split(devices, test_size=0.25, random_state=42)
    train_mask = meta["device_id"].isin(train_dev).values
    test_mask = ~train_mask

    # scale features (fit on train only)
    n_feat = seq_X.shape[-1]
    scaler = StandardScaler()
    scaler.fit(seq_X[train_mask].reshape(-1, n_feat))

    def scale(x):
        shape = x.shape
        return scaler.transform(x.reshape(-1, n_feat)).reshape(shape).astype(np.float32)

    X_train = torch.tensor(scale(seq_X[train_mask]))
    X_test = torch.tensor(scale(seq_X[test_mask]))
    y_train = torch.tensor(y[train_mask], dtype=torch.float32)
    y_test_np = y[test_mask]

    pos_weight = torch.tensor([(y_train == 0).sum().item() / max(1, (y_train == 1).sum().item())])

    model = BiGRUAttention(n_features=n_feat)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    batch_size = 64
    n_train = X_train.shape[0]
    epochs = 12

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        print(f"Epoch {epoch:2d}/{epochs}  loss={epoch_loss / n_train:.4f}")

    model.eval()
    with torch.no_grad():
        logits, _ = model(X_test)
        proba = torch.sigmoid(logits).numpy()
    preds = (proba >= 0.5).astype(int)

    print("\n=== BiGRU-Attention Failure Prediction ===")
    print(f"ROC-AUC : {roc_auc_score(y_test_np, proba):.4f}")
    print(f"PR-AUC  : {average_precision_score(y_test_np, proba):.4f}")
    print(classification_report(y_test_np, preds, digits=3))

    models_dir = BASE / "models"
    models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / "bigru_attention.pt")
    joblib.dump(scaler, models_dir / "bigru_scaler.joblib")
    joblib.dump({"n_features": n_feat, "hidden_dim": 32}, models_dir / "bigru_config.joblib")
    print(f"Saved model to {models_dir / 'bigru_attention.pt'}")


if __name__ == "__main__":
    main()
