"""P1-06 LSTM 滑窗序列预测（PyTorch，Mac CPU 规模）。

结构：LSTM(hidden=64, layers=2) → 最后时间步 → FC → sigmoid
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from models.common import MODELS_DIR, RESULTS_DIR, eval_classification, get_device, save_json, to_numpy


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)              # (B, T, H)
        return self.fc(out[:, -1, :]).squeeze(-1)   # (B,)


def train_lstm(x_train: np.ndarray, y_train: np.ndarray,
               x_val: np.ndarray, y_val: np.ndarray,
               epochs: int = 50, batch_size: int = 32, lr: float = 1e-3,
               patience: int = 5, model_path: Path | None = None,
               device: torch.device | None = None) -> LSTMClassifier:
    device = device or get_device()
    input_size = x_train.shape[2]
    model = LSTMClassifier(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, bad_epochs = float("inf"), None, 0

    for epoch in tqdm(range(1, epochs + 1), desc="LSTM 训练"):
        model.train()
        perm = torch.randperm(len(xt))
        epoch_loss = 0.0
        for i in range(0, len(xt), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            logits = model(xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        train_loss = epoch_loss / len(xt)

        model.eval()
        with torch.no_grad():
            val_logits = model(xv)
            val_loss = loss_fn(val_logits, yv).item()
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))

        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"早停于 epoch {epoch}（val_loss 连续 {patience} 轮未改善）")
                break
        if epoch % 10 == 0:
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    path = model_path or MODELS_DIR / "lstm.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "input_size": input_size, "history": history}, path)
    print(f"LSTM 模型已保存: {path}")
    return model


def predict_proba(model: LSTMClassifier, x: np.ndarray,
                  batch_size: int = 64, device: torch.device | None = None) -> np.ndarray:
    device = device or get_device()
    model.to(device).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = torch.tensor(x[i:i + batch_size], dtype=torch.float32, device=device)
            probs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(probs)


def run_lstm_pipeline(x_train, y_train, x_val, y_val, x_test, y_test,
                      dates_test, closes_test, save_dir: Path | None = None) -> dict:
    """完整流程：训练 → 评估 → 预测落盘 → 指标保存。"""
    from models.common import save_predictions

    model = train_lstm(x_train, y_train, x_val, y_val)
    prob = predict_proba(model, x_test)
    metrics = eval_classification(y_test, prob)
    print(f"[LSTM test] accuracy={metrics['accuracy']} auc={metrics['auc']}")

    save_json(metrics, (save_dir or RESULTS_DIR) / "lstm_metrics.json")
    save_predictions(dates_test, y_test, prob, closes_test,
                     (save_dir or RESULTS_DIR) / "lstm_predictions.csv")
    return metrics


def load_lstm(model_path: Path | None = None, device: torch.device | None = None) -> LSTMClassifier:
    path = model_path or MODELS_DIR / "lstm.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = LSTMClassifier(ckpt["input_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device or get_device()).eval()
    return model
