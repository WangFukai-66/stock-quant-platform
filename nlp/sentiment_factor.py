"""P2B-02 / P2B-03 情绪因子回测与新闻事件驱动分析。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nlp.sentiment import analyze_news


def build_sentiment_factor(news_df: pd.DataFrame, use_deep: bool = False,
                           lookback_days: int = 5) -> pd.DataFrame:
    """新闻 → 日频情绪因子。

    因子：
      情绪均值   当日新闻情绪分均值
      情绪趋势   情绪均值 5 日变化（今-5日前）
      极端情绪占比  近 5 日极端（|分|>0.5）新闻占比
      舆情热度   当日新闻数
    """
    if news_df is None or news_df.empty:
        return pd.DataFrame(columns=["日期", "情绪均值", "情绪趋势", "极端情绪占比", "舆情热度"])
    scored, daily = analyze_news(news_df, use_deep=use_deep)
    if daily.empty:
        return pd.DataFrame(columns=["日期", "情绪均值", "情绪趋势", "极端情绪占比", "舆情热度"])

    daily["情绪趋势"] = daily["情绪均值"].diff(lookback_days)
    scored["极端"] = scored["情绪分"].abs() > 0.5
    extreme = scored.dropna(subset=["发布时间"]).groupby(
        scored["发布时间"].dt.date)["极端"].mean().reset_index()
    extreme.columns = ["日期", "极端情绪占比"]
    out = daily.merge(extreme, on="日期", how="left").fillna(0)
    return out.rename(columns={"新闻数": "舆情热度"})


def merge_sentiment_with_quote(factor: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
    """情绪因子与行情对齐（按日期）。"""
    if factor is None or factor.empty:
        return pd.DataFrame()
    f = factor.copy()
    f["日期"] = pd.to_datetime(f["日期"])
    o = ohlc.copy()
    o["日期"] = pd.to_datetime(o["日期"])
    merged = o.merge(f, on="日期", how="left")
    for c in ("情绪均值", "情绪趋势", "极端情绪占比", "舆情热度"):
        merged[c] = merged[c].fillna(0)
    return merged


def sentiment_backtest(symbol: str, with_sentiment: bool, use_deep: bool = False):
    """对比 加入/不加入情绪因子的 XGBoost 验证集 AUC（增量收益检验）。"""
    from data.fetcher import fetch_news
    from data.quote import get_history
    from factors.process import build_dataset, split_temporal
    from models.xgboost_model import XGBBaseline
    from scripts.pipeline import prepare_tabular

    if not with_sentiment:
        train, val, test, _ = prepare_tabular(symbol)
        model = XGBBaseline()
        model.fit(train, val, verbose=False)
        return {"mode": "无情绪因子", "val_auc": model._evaluate(val, "val")["auc"]}

    # 有情绪因子：因子表 + 情绪因子合并后重训
    ohlc = get_history(symbol, period="daily")
    news = fetch_news(symbol)
    factor = build_sentiment_factor(news, use_deep=use_deep)
    merged = merge_sentiment_with_quote(factor, ohlc)

    ds = build_dataset(ohlc)
    ds = ds.merge(merged[["日期", "情绪均值", "情绪趋势", "极端情绪占比", "舆情热度"]],
                  on="日期", how="left").fillna(0)
    train, val, test = split_temporal(ds)
    extra = ["情绪均值", "情绪趋势", "极端情绪占比", "舆情热度"]
    from factors.process import FACTOR_COLS, standardize
    train, val, test, _ = standardize(train, val, test, cols=FACTOR_COLS + extra)

    model = XGBBaseline()
    model.feature_names = list(FACTOR_COLS) + extra
    model.fit(train, val, verbose=False)
    return {"mode": "含情绪因子", "val_auc": model._evaluate(val, "val")["auc"],
            "importance": model.importance(top=10).to_dict("records")}


def event_study(news_df: pd.DataFrame, minute_df: pd.DataFrame,
                pre_minutes: int = 60, post_minutes: int = 60) -> pd.DataFrame:
    """P2B-03 事件研究：新闻发布前后窗口的收益与成交量偏离。"""
    if news_df is None or news_df.empty or minute_df is None or minute_df.empty:
        return pd.DataFrame()
    news = news_df.dropna(subset=["发布时间"]).sort_values("发布时间")
    m = minute_df.copy()
    m["日期"] = pd.to_datetime(m["日期"])
    m = m.set_index("日期")
    base_vol = m["成交量"].rolling(60, min_periods=10).mean()

    rows = []
    for _, n in news.iterrows():
        t = n["发布时间"]
        win = m[(m.index >= t - pd.Timedelta(minutes=pre_minutes)) &
                (m.index <= t + pd.Timedelta(minutes=post_minutes))]
        if len(win) < 10:
            continue
        pre = win[win.index <= t]["收盘"]
        post = win[win.index > t]["收盘"]
        if len(pre) < 5 or len(post) < 5:
            continue
        ret = (post.iloc[-1] / pre.iloc[-1] - 1) * 100
        vol_ratio = (win["成交量"].mean() / base_vol.reindex(win.index).mean())
        rows.append({"时间": t, "标题": str(n["标题"])[:40],
                     "窗口收益%": round(float(ret), 2),
                     "量比(窗口/60日)": round(float(vol_ratio), 2),
                     "高影响": bool(abs(ret) > 2 or vol_ratio > 3)})
    return pd.DataFrame(rows)


def write_sentiment_report(with_result: dict, without_result: dict) -> str:
    """生成情绪因子增量收益报告。"""
    lines = ["# 情绪因子增量收益报告", "",
             "## 对比结果", "",
             f"- 无情绪因子: 验证集 AUC = {without_result['val_auc']}",
             f"- 含情绪因子: 验证集 AUC = {with_result['val_auc']}", ""]
    diff = with_result["val_auc"] - without_result["val_auc"]
    if diff > 0.005:
        lines.append(f"结论: 情绪因子带来 +{diff:.4f} AUC 增量，建议纳入融合特征。")
    elif diff < -0.005:
        lines.append(f"结论: 情绪因子使 AUC 下降 {diff:.4f}，当前粒度下未体现增量，"
                     "建议改用深度模型情绪分或更长回看窗口。")
    else:
        lines.append("结论: 情绪因子对 AUC 影响不显著（|Δ|<0.005），样本期内无明显增量。")
    lines += ["", "## 情绪因子重要性（Top10）", ""]
    for r in with_result.get("importance", []):
        lines.append(f"- {r['因子']}: {r['重要性']:.4f}")
    return "\n".join(lines)
