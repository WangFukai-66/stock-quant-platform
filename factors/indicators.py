"""P1-04 技术指标计算：MA/MACD/RSI/KDJ/BOLL/ATR。

约定：所有计算仅使用当日及以前的数据（不含未来数据）。
输入 DataFrame 需含列：开盘/收盘/最高/最低/成交量。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=1).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """返回 dif/dea/hist 三列。"""
    dif = ema(close, fast) - ema(close, slow)
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2
    return pd.DataFrame({"MACD_DIF": dif, "MACD_DEA": dea, "MACD_HIST": hist})


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50).rename("RSI")


def kdj(df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    low_n = df["最低"].rolling(n, min_periods=1).min()
    high_n = df["最高"].rolling(n, min_periods=1).max()
    rsv = (df["收盘"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"KDJ_K": k, "KDJ_D": d, "KDJ_J": j})


def boll(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window, min_periods=1).mean()
    std = close.rolling(window, min_periods=1).std().fillna(0)
    return pd.DataFrame({"BOLL_MID": mid, "BOLL_UP": mid + k * std, "BOLL_LOW": mid - k * std})


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["收盘"].shift(1)
    tr = pd.concat([df["最高"] - df["最低"],
                    (df["最高"] - prev_close).abs(),
                    (df["最低"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在行情 DataFrame 上追加全部技术指标列。"""
    out = df.copy()
    close = out["收盘"]
    for w in (5, 10, 20, 60):
        out[f"MA{w}"] = ma(close, w)
    out = pd.concat([out, macd(close), rsi(close), kdj(out), boll(close)], axis=1)
    out["ATR"] = atr(out)
    return out
