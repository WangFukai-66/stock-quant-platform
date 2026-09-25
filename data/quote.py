"""P1-01 行情统一接口：日K/分钟线 + 前复权 + 衍生列。"""
from __future__ import annotations

import pandas as pd

from data.fetcher import fetch_daily, fetch_minute


def get_history(symbol: str, period: str = "daily", adjust: str = "qfq",
                start: str = "20150101", end: str | None = None) -> pd.DataFrame:
    """统一行情入口。

    period: daily / min1 / min5 / min15 / min30 / min60
    adjust: qfq 前复权 / hfq 后复权 / "" 不复权（分钟线不支持复权）
    """
    if period == "daily":
        df = fetch_daily(symbol, adjust=adjust, start=start, end=end)
    elif period.startswith("min"):
        m = period.replace("min", "")
        df = fetch_minute(symbol, period=m)
    else:
        raise ValueError(f"不支持的周期: {period}")

    if df is None or df.empty:
        return pd.DataFrame(columns=["日期", "开盘", "收盘", "最高", "最低", "成交量"])
    df = df.copy()

    # 衍生列
    df["涨跌幅"] = df["收盘"].pct_change() * 100
    df["振幅"] = (df["最高"] - df["最低"]) / df["收盘"].shift(1) * 100
    return df.reset_index(drop=True)


def resample_minute(df: pd.DataFrame, rule: str = "15min") -> pd.DataFrame:
    """分钟线聚合为更大周期。"""
    out = df.set_index("日期").resample(rule).agg(
        开盘=("开盘", "first"), 最高=("最高", "max"), 最低=("最低", "min"),
        收盘=("收盘", "last"), 成交量=("成交量", "sum"), 成交额=("成交额", "sum"))
    return out.dropna().reset_index()
