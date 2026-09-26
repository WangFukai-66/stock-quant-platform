"""P2A-01 Transformer 时序模型（纯 PyTorch，Mac CPU 规模）。

结构：因子投影(d_model) → CLS token + 位置编码 → 2 层 TransformerEncoder → CLS → FC → sigmoid
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.common import MODELS_DIR, RESULTS_DIR, eval_classification, get_device, save_json


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                                    # (B, T, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)                      # (B, T+1, d_model)
        x = self.pos_enc(x)
        out = self.encoder(x)[:, 0, :]                      # CLS 输出
        return self.head(out).squeeze(-1)


def train_transformer(x_train, y_train, x_val, y_val, epochs: int = 50,
                      batch_size: int = 32, lr: float = 1e-3, patience: int = 5,
                      model_path: Path | None = None, device=None,
                      progress_cb=None) -> TimeSeriesTransformer:
    """progress_cb(epoch, epochs)：每个 epoch 结束回调，用于外部进度展示。"""
    device = device or get_device()
    model = TimeSeriesTransformer(x_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in tqdm(range(1, epochs + 1), desc="Transformer 训练"):
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        if progress_cb is not None:
            progress_cb(epoch, epochs)
        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"早停于 epoch {epoch}")
                break
        if epoch % 10 == 0:
            print(f"epoch {epoch}: train_loss={total/len(xt):.4f} val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    path = model_path or MODELS_DIR / "transformer.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "input_size": x_train.shape[2]}, path)
    print(f"Transformer 模型已保存: {path}")
    return model


def predict_proba(model: TimeSeriesTransformer, x: np.ndarray,
                  batch_size: int = 64, device=None) -> np.ndarray:
    device = device or get_device()
    model.to(device).eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            batch = torch.tensor(x[i:i + batch_size], dtype=torch.float32, device=device)
            probs.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(probs)


def load_transformer(model_path: Path | None = None, device=None) -> TimeSeriesTransformer:
    path = model_path or MODELS_DIR / "transformer.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = TimeSeriesTransformer(ckpt["input_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device or get_device()).eval()
    return model


def run_transformer_pipeline(x_train, y_train, x_val, y_val, x_test, y_test,
                             dates_test, closes_test) -> dict:
    from models.common import save_predictions
    model = train_transformer(x_train, y_train, x_val, y_val)
    prob = predict_proba(model, x_test)
    metrics = eval_classification(y_test, prob)
    print(f"[Transformer test] accuracy={metrics['accuracy']} auc={metrics['auc']}")
    save_json(metrics, RESULTS_DIR / "transformer_metrics.json")
    save_predictions(dates_test, y_test, prob, closes_test,
                     RESULTS_DIR / "transformer_predictions.csv")
    return metrics
