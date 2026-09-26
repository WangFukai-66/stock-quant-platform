"""看板公共组件：市场状态四象限、K线信号标注（torch-free，云端可安全引用）。

market_state 与 models/fusion.py 中逻辑一致，此处为展示用副本
（models/fusion.py 顶层依赖 torch，看板运行路径不得引用）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STATE_NAMES = {
    0: "低波动·震荡",
    1: "低波动·趋势",
    2: "高波动·震荡",
    3: "高波动·趋势",
}

SIGNAL_STYLE = {
    "买入": dict(symbol="triangle-up", color="#ef4444", size=11),
    "卖出": dict(symbol="triangle-down", color="#22c55e", size=11),
    "风险": dict(symbol="x", color="#f59e0b", size=9),
}


def market_state(close: pd.Series, vol_window: int = 20) -> pd.Series:
    """市场状态四象限：高/低波动 × 趋势/震荡 → 0~3（展示用）。"""
    ret = close.pct_change()
    vol = ret.rolling(vol_window).std()
    vol_q = vol.rolling(252, min_periods=60).apply(
        lambda x: (x.iloc[-1] >= np.quantile(x, 0.7)), raw=False)
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    slope = ma20.diff(5)
    trend = ((close > ma60) & (slope > 0)) | ((close < ma60) & (slope < 0))
    high_vol = vol_q.fillna(False).astype(bool)
    state = high_vol.astype(int) * 2 + trend.astype(int)
    return state.fillna(0).astype(int)


def kline_with_signals(df: pd.DataFrame, signals: pd.DataFrame | None,
                       days: int = 120, title: str = "K线与预警信号") -> go.Figure:
    """K线 + MA + 成交量，叠加买卖/风险信号标注（红涨绿跌）。"""
    d = df.tail(days)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03,
                        subplot_titles=(title, "成交量"))
    fig.add_trace(go.Candlestick(x=d["日期"], open=d["开盘"], high=d["最高"],
                                 low=d["最低"], close=d["收盘"], name="K线",
                                 increasing_line_color="red", decreasing_line_color="green"),
                  row=1, col=1)
    for ma_col, color in [("MA5", "#f59e0b"), ("MA20", "#3b82f6"), ("MA60", "#8b5cf6")]:
        if ma_col in d.columns:
            fig.add_trace(go.Scatter(x=d["日期"], y=d[ma_col], name=ma_col,
                                     line=dict(width=1, color=color)), row=1, col=1)
    colors = ["red" if c >= o else "green" for o, c in zip(d["开盘"], d["收盘"])]
    fig.add_trace(go.Bar(x=d["日期"], y=d["成交量"], name="成交量",
                         marker_color=colors), row=2, col=1)

    if signals is not None and not signals.empty:
        sig = signals[signals["日期"].isin(d["日期"])]
        for direction, style in SIGNAL_STYLE.items():
            sub = sig[sig["方向"] == direction]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["日期"], y=sub["触发价"], mode="markers",
                name=f"{direction}信号", marker=style,
                text=[f"{t}<br>{n}" for t, n in zip(sub["类型"], sub["说明"])],
                hoverinfo="text+x"), row=1, col=1)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h"))
    return fig
