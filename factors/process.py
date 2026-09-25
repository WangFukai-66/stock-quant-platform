"""P1-04 特征工程：量价因子、标准化、标签生成。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factors.indicators import add_all_indicators

# 模型使用的因子列（不含标签与行情原始列）
FACTOR_COLS = [
    "MA5", "MA10", "MA20", "MA60",
    "MACD_DIF", "MACD_DEA", "MACD_HIST",
    "RSI", "KDJ_K", "KDJ_D", "KDJ_J",
    "BOLL_MID", "BOLL_UP", "BOLL_LOW", "ATR",
    "量比", "动量5", "动量20", "波动率20", "振幅", "涨跌幅",
    "价格位置", "成交额",
]

PRICE_COLS = ["开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "换手率"]


def add_price_factors(df: pd.DataFrame) -> pd.DataFrame:
    """量价因子（仅用历史信息）。"""
    out = df.copy()
    close = out["收盘"]
    vol = out["成交量"]
    # 量比：当日成交量 / 近5日均量（含当日会造成轻微稀释，改用前5日均量更严格）
    out["量比"] = vol / vol.shift(1).rolling(5, min_periods=1).mean()
    out["动量5"] = close.pct_change(5) * 100
    out["动量20"] = close.pct_change(20) * 100
    out["波动率20"] = close.pct_change().rolling(20, min_periods=2).std() * 100
    out["振幅"] = (out["最高"] - out["最低"]) / close.shift(1) * 100
    out["涨跌幅"] = close.pct_change() * 100
    # 价格在近 60 日内的位置：(close - min60) / (max60 - min60)
    min60 = close.rolling(60, min_periods=2).min()
    max60 = close.rolling(60, min_periods=2).max()
    out["价格位置"] = (close - min60) / (max60 - min60).replace(0, np.nan)
    return out


def build_dataset(df: pd.DataFrame, warmup: int = 60) -> pd.DataFrame:
    """行情 -> 因子表 + 标签。

    标签：
      label      次日涨跌（1 涨 / 0 跌），用于分类
      ret_next   次日收益率（%），用于回归与回测
    返回列：日期 + 因子 + 标签 + 收盘（供回测使用）
    """
    out = add_all_indicators(df)
    out = add_price_factors(out)
    out["label"] = (out["收盘"].shift(-1) > out["收盘"]).astype(int)
    out["ret_next"] = out["收盘"].pct_change().shift(-1) * 100
    out = out.dropna(subset=["收盘"]).iloc[warmup:]      # 截断指标预热期
    cols = [c for c in FACTOR_COLS if c in out.columns]
    out = out.dropna(subset=cols).reset_index(drop=True)
    return out


def split_temporal(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """严格按时间顺序切分，无 shuffle（防未来函数）。"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    return train, val, test


def standardize(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                cols: list[str] | None = None) -> tuple:
    """z-score 标准化：均值/标准差只用训练集统计量。"""
    cols = cols or FACTOR_COLS
    mu = train[cols].mean()
    sd = train[cols].std().replace(0, 1)
    def _z(x):
        y = x.copy()
        y[cols] = (y[cols] - mu) / sd
        return y
    return _z(train), _z(val), _z(test), (mu, sd)


def make_windows(df: pd.DataFrame, window: int = 60, cols: list[str] | None = None):
    """构造滑窗序列 (N, window, F) 及对应标签/日期/收盘价。"""
    cols = cols or FACTOR_COLS
    x = df[cols].values
    xs, ys, dates, closes = [], [], [], []
    for i in range(window, len(df)):
        xs.append(x[i - window:i])
        ys.append(df["label"].iloc[i])
        dates.append(df["日期"].iloc[i])
        closes.append(df["收盘"].iloc[i])
    return (np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int64),
            np.array(dates), np.array(closes, dtype=np.float64))
