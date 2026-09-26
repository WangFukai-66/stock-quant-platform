"""基准对比：标的买入持有 + 沪深300 指数，用于回测超额收益展示。

仅依赖 pandas，可被看板安全引用（不引入 torch/transformers）。
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def buy_hold_equity(ohlc: pd.DataFrame, initial_cash: float = 1_000_000.0) -> pd.Series:
    """标的自身买入持有资金曲线（首日开盘全仓买入，按收盘计值）。"""
    if ohlc is None or ohlc.empty:
        return pd.Series(dtype=float)
    first_open = float(ohlc.iloc[0]["开盘"])
    close = ohlc.set_index("日期")["收盘"].astype(float)
    shares = int(initial_cash // (first_open * 100)) * 100
    if shares == 0:
        return pd.Series(initial_cash, index=close.index)
    cash = initial_cash - shares * first_open
    return cash + shares * close


def index_equity(index_code: str = "sh000300", start: str = "20150101") -> pd.Series | None:
    """沪深300 归一化净值曲线（基准=1）。接口失败返回 None（降级为仅买入持有对比）。"""
    try:
        from data.fetcher import fetch_index_daily
        idx = fetch_index_daily(index_code, start=start)
    except Exception as exc:  # 云端/离线环境接口不可达
        logger.warning("指数行情获取失败(%s)，基准对比降级为买入持有", exc)
        return None
    if idx is None or idx.empty or "收盘" not in idx.columns:
        return None
    close = idx.set_index("日期")["收盘"].astype(float)
    return close / close.iloc[0]


def excess_metrics(strategy_equity: pd.Series, benchmark: pd.Series,
                   initial_cash: float = 1_000_000.0) -> dict:
    """策略 vs 基准：超额收益、超额年化（按共同区间对齐）。"""
    if benchmark is None or benchmark.empty:
        return {}
    s = strategy_equity / strategy_equity.iloc[0]
    common = s.index.intersection(benchmark.index)
    if len(common) < 2:
        return {}
    s, b = s.loc[common], benchmark.loc[common]
    excess = float((s.iloc[-1] / b.iloc[-1]) - 1) * 100
    years = max(len(common) / 252, 1e-9)
    annual_excess = float(((s.iloc[-1] / b.iloc[-1]) ** (1 / years) - 1) * 100)
    return {"超额收益": round(excess, 2), "年化超额": round(annual_excess, 2)}
