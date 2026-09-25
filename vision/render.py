"""P2C-01 K线图标准化渲染 + 合成形态数据生成。

渲染：64×64 RGB、统一灰度背景、含 MA5/MA20、无坐标轴文字（防模型依赖文字）。
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

IMG_SIZE = 64


def render_candlestick(df: pd.DataFrame, size: int = IMG_SIZE) -> Image.Image:
    """近 60 根 K 线渲染为 (size, size) RGB 图像。"""
    d = df.tail(60).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")
    x = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        color = "#e74c3c" if r["收盘"] >= r["开盘"] else "#2ecc71"
        ax.plot([i, i], [r["最低"], r["最高"]], color=color, linewidth=0.6)
        ax.add_patch(plt.Rectangle((i - 0.35, min(r["开盘"], r["收盘"])), 0.7,
                                   max(abs(r["收盘"] - r["开盘"]), 0.02),
                                   facecolor=color, edgecolor=color))
    ma5 = d["收盘"].rolling(5).mean()
    ma20 = d["收盘"].rolling(20).mean()
    ax.plot(x, ma5, color="#f39c12", linewidth=0.6)
    ax.plot(x, ma20, color="#3498db", linewidth=0.6)
    ax.axis("off")
    ax.set_xlim(-1, len(d))
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB").resize((size, size))


def render_batch(ohlc: pd.DataFrame, window: int = 60) -> list[Image.Image]:
    """按滑窗批量渲染，返回 [img_t(近window日), ...] 与窗口日期对齐。"""
    images, dates = [], []
    for i in range(window, len(ohlc) + 1):
        images.append(render_candlestick(ohlc.iloc[i - window:i]))
        dates.append(ohlc.iloc[i - 1]["日期"])
    return images, dates


# ---------------------------------------------------------------- 合成形态 ----

PATTERNS = ["头肩顶", "双底", "上升三角", "箱体", "突破"]


def _random_walk(n: int, drift: float = 0.0, vol: float = 0.015, seed=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, vol, n)
    return 100 * np.exp(np.cumsum(ret))


def _to_ohlc(close: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng()
    o = close * (1 + rng.normal(0, 0.004, len(close)))
    h = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 0.006, len(close))))
    l = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 0.006, len(close))))
    return pd.DataFrame({"日期": pd.date_range(end=pd.Timestamp.today(), periods=len(close)),
                         "开盘": o, "收盘": close, "最高": h, "最低": l,
                         "成交量": rng.uniform(1e5, 5e5, len(close))})


def synthesize_patterns(pattern: str, n: int = 200, length: int = 60) -> list[pd.DataFrame]:
    """合成 K 线形态样本（用于 CNN 训练数据增强）。"""
    out = []
    for _ in range(n):
        base = _random_walk(length)
        idx = np.arange(length)
        if pattern == "头肩顶":
            shape = np.exp(-((idx - length * 0.6) ** 2) / (length * 0.12) ** 2) * 0.25
            shape += np.exp(-((idx - length * 0.45) ** 2) / (length * 0.05) ** 2) * 0.12
            shape += np.exp(-((idx - length * 0.75) ** 2) / (length * 0.05) ** 2) * 0.12
            close = base * (1 + shape) * (1 - 0.25 * idx / length)
        elif pattern == "双底":
            shape = -np.exp(-((idx - length * 0.35) ** 2) / (length * 0.1) ** 2) * 0.3
            shape += -np.exp(-((idx - length * 0.65) ** 2) / (length * 0.1) ** 2) * 0.3
            close = base * (1 + shape) * (1 + 0.3 * idx / length)
        elif pattern == "上升三角":
            close = base * (1 + 0.5 * idx / length) * (1 - 0.15 * np.maximum(0, idx - length * 0.5) / length)
        elif pattern == "箱体":
            mid = base.mean()
            close = base * 0.3 + mid * 0.7 + np.sin(idx / 3) * mid * 0.02
        elif pattern == "突破":
            plateau = base * 0.98
            jump = np.where(idx > length * 0.75, 0.25, 0.0)
            close = plateau * (1 + jump)
        else:
            close = base
        out.append(_to_ohlc(close))
    return out


def build_synthetic_dataset(n_per_class: int = 200) -> tuple[list[Image.Image], list[int]]:
    """生成 (图像, 标签) 数据集，标签与 PATTERNS 索引对应。"""
    images, labels = [], []
    for label, p in enumerate(PATTERNS):
        for df in synthesize_patterns(p, n=n_per_class):
            images.append(render_candlestick(df))
            labels.append(label)
    return images, labels
