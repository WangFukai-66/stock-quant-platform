"""P1-08 规则预警引擎：买卖点信号、风险预警、仓位建议。

信号在当日收盘后基于当日数据触发（次日可执行）。
"""
from __future__ import annotations

import pandas as pd

from factors.indicators import add_all_indicators

# 预警类型
GOLDEN_CROSS = "金叉(MA5上穿MA20)"      # 买入
DEATH_CROSS = "死叉(MA5下穿MA20)"       # 卖出
LIMIT_UP = "涨停触板"                    # 风险提示
LIMIT_DOWN = "跌停触板"                  # 风险提示
VOLUME_SURGE = "放量异动"                # 关注
MACD_BULL_DIV = "MACD底背离"             # 买入
MACD_BEAR_DIV = "MACD顶背离"             # 卖出
DRAWDOWN_ALERT = "回撤预警"              # 风险


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """输入日K（含日期/开盘/收盘/最高/最低/成交量），输出当日全部信号。"""
    # 防御：看板可能传入已带指标列的 DataFrame，先剔除再重算，避免 concat 产生重复列名
    indicator_cols = ["MA5", "MA10", "MA20", "MA60", "MACD_DIF", "MACD_DEA", "MACD_HIST",
                      "RSI", "KDJ_K", "KDJ_D", "KDJ_J", "BOLL_MID", "BOLL_UP", "BOLL_LOW", "ATR"]
    df = df.drop(columns=[c for c in indicator_cols if c in df.columns])
    out = add_all_indicators(df).copy()
    out["量比"] = out["成交量"] / out["成交量"].shift(1).rolling(5, min_periods=1).mean()
    out["涨跌幅"] = out["收盘"].pct_change() * 100

    signals = []
    for i in range(1, len(out)):
        row = out.iloc[i]
        prev = out.iloc[i - 1]
        date = row["日期"]
        close = float(row["收盘"])
        prev_close = float(prev["收盘"])
        pct = float(row["涨跌幅"]) if pd.notna(row["涨跌幅"]) else 0.0

        def add(sig_type: str, direction: str, note: str, action: str):
            signals.append({
                "日期": date, "类型": sig_type, "方向": direction,
                "触发价": round(close, 2), "说明": note, "操作建议": action,
            })

        # 均线金叉/死叉
        if _cross_up(out["MA5"], out["MA20"]).iloc[i]:
            add(GOLDEN_CROSS, "买入", f"MA5({row['MA5']:.2f}) 上穿 MA20({row['MA20']:.2f})", "可建仓")
        if _cross_down(out["MA5"], out["MA20"]).iloc[i]:
            add(DEATH_CROSS, "卖出", f"MA5({row['MA5']:.2f}) 下穿 MA20({row['MA20']:.2f})", "可减仓")

        # 涨跌停触板（±10% 主板）
        if pct >= 9.9:
            add(LIMIT_UP, "风险", "触及涨停板，追高需谨慎", "不追高")
        if pct <= -9.9:
            add(LIMIT_DOWN, "风险", "触及跌停板，注意流动性风险", "不抄底")

        # 放量异动
        if row["量比"] > 2:
            direction = "买入" if pct > 0 else "卖出"
            add(VOLUME_SURGE, direction, f"量比 {row['量比']:.1f}（>2）且涨跌 {pct:.2f}%", "重点观察")

        # MACD 背离（近 20 日简化判断：价创新高/新低但 DIF 未同步）
        if i >= 20:
            win = out.iloc[i - 20:i + 1]
            if row["收盘"] == win["收盘"].max() and row["MACD_DIF"] < win["MACD_DIF"].max():
                add(MACD_BEAR_DIV, "卖出", "价格创 20 日新高但 MACD 未创新高", "警惕回调")
            if row["收盘"] == win["收盘"].min() and row["MACD_DIF"] > win["MACD_DIF"].min():
                add(MACD_BULL_DIV, "买入", "价格创 20 日新低但 MACD 未创新低", "关注反弹")

        # 回撤预警：从近 20 日高点回撤超 10%
        high20 = out["收盘"].iloc[i - 20:i + 1].max()
        if high20 > 0 and (high20 - close) / high20 >= 0.10:
            add(DRAWDOWN_ALERT, "风险", f"自 20 日高点 {high20:.2f} 回撤 {(high20-close)/high20*100:.1f}%", "控制仓位")

    return pd.DataFrame(signals)


def latest_signals(df: pd.DataFrame, asof_date=None) -> pd.DataFrame:
    """取最新一日的信号。"""
    sig = generate_signals(df)
    if sig.empty:
        return sig
    latest = asof_date or sig["日期"].max()
    return sig[sig["日期"] == latest]


def position_advice(df: pd.DataFrame) -> dict:
    """仓位建议：统计当日信号方向 → 满仓/半仓/降仓/观望。"""
    sig = latest_signals(df)
    if sig.empty:
        return {"日期": str(df["日期"].iloc[-1]), "建议仓位": "观望", "依据": "当日无触发信号"}

    buy_n = int((sig["方向"] == "买入").sum())
    sell_n = int((sig["方向"] == "卖出").sum())
    risk_n = int((sig["方向"] == "风险").sum())

    if buy_n >= 2 and risk_n == 0:
        advice, reason = "满仓", f"买入信号 {buy_n} 条，无风险信号"
    elif buy_n >= 1 and risk_n == 0:
        advice, reason = "半仓", f"买入信号 {buy_n} 条"
    elif sell_n >= 1 or risk_n >= 2:
        advice, reason = "降仓", f"卖出 {sell_n} 条 / 风险 {risk_n} 条"
    elif risk_n == 1:
        advice, reason = "轻仓", "存在 1 条风险信号"
    else:
        advice, reason = "观望", "信号方向不明确"

    return {"日期": str(df["日期"].iloc[-1]), "建议仓位": advice, "依据": reason,
            "买入信号": buy_n, "卖出信号": sell_n, "风险信号": risk_n}
