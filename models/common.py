"""公共工具：设备抽象、评估指标、结果落盘。

统一 device 抽象，便于 Mac CPU 与云端 GPU（Colab/AutoDL）迁移。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Mac 上 torch 自带的 Intel OpenMP 与 xgboost 的 LLVM libomp 冲突会导致训练 segfault，
# 统一单线程规避（小规模模型训练速度影响可忽略；云端 Linux 同样安全）。
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"


def get_device() -> torch.device:
    """CPU / MPS(Mac) / CUDA 自动选择。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def eval_classification(y_true, y_prob) -> dict:
    """分类评估：accuracy/AUC/混淆矩阵。"""
    y_true = to_numpy(y_true)
    y_prob = to_numpy(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "n_samples": int(len(y_true)),
        "positive_ratio": round(float(y_true.mean()), 4),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"已保存: {path}")


def save_predictions(dates, y_true, y_prob, closes, path: Path) -> None:
    """预测结果落盘（回测引擎输入格式）。"""
    import pandas as pd
    df = pd.DataFrame({
        "日期": dates, "真实标签": y_true, "预测概率": np.round(y_prob, 4), "收盘": closes,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"已保存预测: {path}")
