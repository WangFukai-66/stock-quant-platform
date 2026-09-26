"""P1-09 Streamlit 六页签看板。

启动: streamlit run app/main.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alert.rules import generate_signals, latest_signals, position_advice
from backtest.engine import BacktestEngine, BacktestConfig
from data.fetcher import fetch_news

# 云端增量拉取偶发文件不同步：fetch_stock_names 缺失时降级为仅格式校验，避免应用崩溃
try:
    from data.fetcher import fetch_stock_names
except ImportError:
    fetch_stock_names = None
from data.finance import get_financial_metrics
from data.quote import get_history
from factors.indicators import add_all_indicators
from nlp.sentiment import analyze_news

st.set_page_config(page_title="A股量化回测与智能交易预警系统", layout="wide")

DEFAULT_SYMBOL = "600519"
DEFAULT_NAME = "贵州茅台"


# ---------------------------------------------------------------- 缓存 ----

@st.cache_data(ttl=3600)
def load_history(symbol: str, period: str, adjust: str, start: str):
    return get_history(symbol, period=period, adjust=adjust, start=start)


@st.cache_data(ttl=3600)
def load_indicators(symbol: str, start: str):
    return add_all_indicators(get_history(symbol, period="daily", start=start))


@st.cache_data(ttl=3600)
def load_news(symbol: str):
    return fetch_news(symbol)


@st.cache_data(ttl=3600)
def load_financial(symbol: str):
    return get_financial_metrics(symbol)


@st.cache_data(ttl=86400)
def load_stock_names() -> dict:
    """全市场股票代码→名称映射（用于输入校验与名称显示，失败返回空字典）。"""
    if fetch_stock_names is None:
        return {}
    return fetch_stock_names()


@st.cache_data(ttl=3600)
def load_signals(symbol: str, start: str):
    df = get_history(symbol, period="daily", start=start)
    return generate_signals(df)


@st.cache_data(ttl=3600)
def load_prediction(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["日期"])


def latest_prediction(symbol: str) -> pd.DataFrame | None:
    """读取该标的最新预测（优先融合，其次 XGBoost/LSTM）。

    预测文件带股票代码后缀（如 fusion_predictions_600519.csv）；
    无后缀的旧版文件视为默认标的 600519 的预测，其他股票不读，避免张冠李戴。
    """
    for name in ("fusion", "xgb", "lstm"):
        for suffix in (f"_{symbol}", "" if symbol == DEFAULT_SYMBOL else None):
            if suffix is None:
                continue
            pred = load_prediction(str(PROJECT_ROOT / "results" / f"{name}_predictions{suffix}.csv"))
            if pred is not None and not pred.empty:
                return pred
    return None


def find_metrics(symbol: str, name: str) -> dict | None:
    """读取模型指标 JSON（优先带代码后缀的文件；无后缀旧版文件仅限默认标的）。"""
    for suffix in (f"_{symbol}", "" if symbol == DEFAULT_SYMBOL else None):
        if suffix is None:
            continue
        p = PROJECT_ROOT / "results" / f"{name}_metrics{suffix}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def trained_hint(symbol: str) -> str:
    """未训练标的的统一提示文案。"""
    return (f"股票 {symbol} 暂无模型预测。当前已训练标的：{DEFAULT_SYMBOL} {DEFAULT_NAME}。"
            f"本地训练命令：python -m scripts.train_all --symbol {symbol}")


# ---------------------------------------------------------------- 图表 ----

def kline_fig(df: pd.DataFrame, title: str = "K线图") -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
                        subplot_titles=(title, "成交量", "MACD"))
    fig.add_trace(go.Candlestick(x=df["日期"], open=df["开盘"], high=df["最高"],
                                 low=df["最低"], close=df["收盘"], name="K线",
                                 increasing_line_color="red", decreasing_line_color="green"),
                  row=1, col=1)
    for ma_col, color in [("MA5", "#f59e0b"), ("MA20", "#3b82f6"), ("MA60", "#8b5cf6")]:
        if ma_col in df.columns:
            fig.add_trace(go.Scatter(x=df["日期"], y=df[ma_col], name=ma_col,
                                     line=dict(width=1, color=color)), row=1, col=1)
    colors = ["red" if c >= o else "green" for o, c in zip(df["开盘"], df["收盘"])]
    fig.add_trace(go.Bar(x=df["日期"], y=df["成交量"], name="成交量",
                         marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["日期"], y=df["MACD_DIF"], name="DIF",
                             line=dict(width=1, color="#f59e0b")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["日期"], y=df["MACD_DEA"], name="DEA",
                             line=dict(width=1, color="#3b82f6")), row=3, col=1)
    fig.add_trace(go.Bar(x=df["日期"], y=df["MACD_HIST"], name="MACD柱",
                         marker_color=["red" if v >= 0 else "green" for v in df["MACD_HIST"]]),
                  row=3, col=1)
    fig.update_layout(height=650, xaxis_rangeslider_visible=False, legend=dict(orientation="h"))
    return fig


# ---------------------------------------------------------------- 页签 ----

def page_home(symbol: str, df: pd.DataFrame, pred: pd.DataFrame | None,
              signals: pd.DataFrame, advice: dict):
    st.header("概览")
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    chg = (last["收盘"] / prev["收盘"] - 1) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新收盘价", f"{last['收盘']:.2f}", f"{chg:+.2f}%")
    c2.metric("最新成交量", f"{last['成交量']/1e4:.1f} 万手")
    if pred is not None and not pred.empty:
        p_last = float(pred.iloc[-1]["预测概率"])
        c3.metric("次日上涨概率", f"{p_last*100:.1f}%",
                  "看涨" if p_last >= 0.55 else ("看跌" if p_last < 0.45 else "中性"))
    else:
        c3.metric("次日上涨概率", "未训练")
    today_sig = len(latest_signals(df))
    c4.metric("今日预警信号", f"{today_sig} 条")

    if pred is None or pred.empty:
        st.caption(trained_hint(symbol))

    st.subheader("仓位建议")
    st.info(f"**{advice['建议仓位']}** — {advice['依据']}（买入 {advice.get('买入信号', 0)} / "
            f"卖出 {advice.get('卖出信号', 0)} / 风险 {advice.get('风险信号', 0)}）")

    st.subheader("近 120 日走势")
    st.plotly_chart(kline_fig(df.tail(120), f"{symbol} 近120日"), width='stretch')


def page_quote(df: pd.DataFrame):
    st.header("行情看板")
    st.plotly_chart(kline_fig(df), width='stretch')
    with st.expander("原始数据（近 30 行）"):
        st.dataframe(df.tail(30), width='stretch')


def page_finance(symbol: str):
    st.header("财报分析")
    fin = load_financial(symbol)
    if fin is None or fin.empty:
        st.warning("暂无财务指标数据（akshare 接口可能未覆盖该标的）")
        return
    show = fin.tail(20)
    st.dataframe(show.style.format(precision=2), width='stretch')
    cols = [c for c in ("ROE", "毛利率", "净利率") if c in show.columns]
    if cols:
        fig = go.Figure()
        for c in cols:
            fig.add_trace(go.Scatter(x=show["日期"], y=show[c], name=c, mode="lines+markers"))
        fig.update_layout(height=400, title="盈利能力趋势")
        st.plotly_chart(fig, width='stretch')


def page_news(symbol: str, use_deep: bool):
    st.header("舆情监控")
    news = load_news(symbol)
    if news is None or news.empty:
        st.warning("暂无新闻数据")
        return
    scored, daily = analyze_news(news, use_deep=use_deep)
    mode = "深度模型(RoBERTa)" if use_deep else "规则词典"
    st.caption(f"情绪分析模式：{mode}")

    pos = int((scored["情绪"] == "正面").sum())
    neg = int((scored["情绪"] == "负面").sum())
    neu = int((scored["情绪"] == "中性").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("正面新闻", pos)
    c2.metric("中性新闻", neu)
    c3.metric("负面新闻", neg)

    st.subheader("新闻列表")
    show = scored[["发布时间", "标题", "来源", "情绪", "情绪分"]].head(50)
    st.dataframe(show, width='stretch')

    if not daily.empty:
        st.subheader("舆情热度与情绪时间轴")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=daily["日期"], y=daily["新闻数"], name="新闻数"), secondary_y=False)
        fig.add_trace(go.Scatter(x=daily["日期"], y=daily["情绪均值"], name="情绪均值",
                                 line=dict(color="red")), secondary_y=True)
        fig.update_layout(height=350)
        st.plotly_chart(fig, width='stretch')


def page_predict(symbol: str):
    st.header("AI 预测")
    st.caption("XGBoost / LSTM / Transformer 融合引擎的次日涨跌概率")
    pred = latest_prediction(symbol)
    if pred is None:
        st.warning(trained_hint(symbol))
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred["日期"], y=pred["预测概率"], name="上涨概率",
                             line=dict(color="#ef4444")))
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                  annotation_text="0.5 分界")
    fig.add_hline(y=0.55, line_dash="dot", line_color="orange",
                  annotation_text="买入阈值 0.55")
    fig.update_layout(height=400, title="预测概率走势（测试集）")
    st.plotly_chart(fig, width='stretch')
    st.dataframe(pred.tail(30), width='stretch')

    # 各模型指标
    st.subheader("模型指标")
    cols = st.columns(3)
    for i, name in enumerate(("xgb", "lstm", "fusion")):
        m = find_metrics(symbol, name)
        if m:
            cols[i].markdown(f"**{name.upper()}**  \n"
                             f"accuracy={m.get('accuracy')}  \nauc={m.get('auc')}")
        else:
            cols[i].markdown(f"**{name.upper()}**  \n未训练")


def page_backtest(symbol: str, threshold: float):
    st.header("量化回测")
    st.caption("严格模拟 A 股规则：T+1、涨跌停不可成交、佣金万2.5、印花税卖出千1、滑点1分")
    pred = latest_prediction(symbol)
    if pred is None:
        st.warning(trained_hint(symbol))
        return
    ohlc = load_history(symbol, "daily", "qfq", "20150101")
    merged = ohlc.merge(pred[["日期", "预测概率"]], on="日期", how="inner")
    if merged.empty:
        st.warning("预测与行情日期无交集")
        return

    cfg = BacktestConfig(threshold=threshold)
    result = BacktestEngine(cfg).run(merged, merged["预测概率"])

    m = result.metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总收益率", f"{m['总收益率']}%")
    c2.metric("年化收益率", f"{m['年化收益率']}%")
    c3.metric("夏普比率", f"{m['夏普比率']}")
    c4.metric("最大回撤", f"{m['最大回撤']}%")
    c5.metric("交易次数", m["交易次数"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05,
                        subplot_titles=("资金曲线", "回撤"))
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity.values,
                             name="资产", line=dict(color="#2563eb")), row=1, col=1)
    fig.add_trace(go.Scatter(x=result.drawdown.index, y=result.drawdown.values * 100,
                             name="回撤%", line=dict(color="#dc2626"), fill="tozeroy"), row=2, col=1)
    fig.update_layout(height=550, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, width='stretch')

    if result.trades is not None and not result.trades.empty:
        st.subheader("交易明细")
        st.dataframe(result.trades.tail(30), width='stretch')


def page_alert(symbol: str):
    st.header("智能预警")
    sig = load_signals(symbol, "20230101")
    if sig is None or sig.empty:
        st.info("近一年无触发信号")
        return
    advice = position_advice(get_history(symbol, period="daily", start="20230101"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("累计信号", len(sig))
    c2.metric("买入信号", int((sig["方向"] == "买入").sum()))
    c3.metric("卖出信号", int((sig["方向"] == "卖出").sum()))
    c4.metric("风险信号", int((sig["方向"] == "风险").sum()))

    st.subheader("当前仓位建议")
    st.info(f"**{advice['建议仓位']}** — {advice['依据']}")

    st.subheader("最近信号")
    st.dataframe(sig.tail(30), width='stretch')


# ---------------------------------------------------------------- 主入口 ----

def main():
    st.sidebar.title("A股量化回测与智能交易预警系统")

    # 股票代码校验（6 位数字）+ 名称显示（接口失败时仅做格式校验）
    symbol = st.sidebar.text_input("股票代码", DEFAULT_SYMBOL).strip()
    if not re.fullmatch(r"\d{6}", symbol):
        st.sidebar.error("请输入 6 位数字股票代码，如 600519 贵州茅台")
        st.stop()
    names = load_stock_names()
    if symbol in names:
        st.sidebar.caption(f"**{names[symbol]}**")
    elif names:
        st.sidebar.error(f"股票代码 {symbol} 不存在，请检查后重新输入")
        st.stop()

    period = st.sidebar.selectbox("行情周期", ["daily", "min5", "min15", "min30", "min60"])
    adjust = st.sidebar.selectbox("复权方式", ["qfq", "hfq", ""], format_func=lambda x: x or "不复权")
    start = st.sidebar.date_input("起始日期", value=pd.to_datetime("2015-01-01")).strftime("%Y%m%d")
    threshold = st.sidebar.slider("回测买入概率阈值", 0.50, 0.80, 0.55, 0.01)

    # 云端（1GB 内存）禁用 RoBERTa 深度情绪模型，避免下载 400MB 模型导致 OOM
    # （新版 Streamlit 在无 secrets.toml 时会抛异常，需防御）
    try:
        deep_disabled = bool(st.secrets.get("DISABLE_DEEP_MODEL", False))
    except Exception:
        deep_disabled = False
    if deep_disabled:
        st.sidebar.caption("云端环境已禁用深度情绪模型（内存限制），自动使用规则词典打分")
        use_deep = False
    else:
        use_deep = st.sidebar.checkbox("舆情深度情绪模型（RoBERTa，首次加载较慢）", value=False)

    page = st.sidebar.radio("功能页签", ["概览", "行情", "财报", "舆情", "AI预测", "量化回测", "智能预警"])

    with st.spinner("加载数据中..."):
        daily_df = load_indicators(symbol, start)
        if daily_df is None or daily_df.empty:
            st.error(f"未获取到 {symbol} 的行情数据，请检查股票代码")
            return

    pred = latest_prediction(symbol)
    if page in ("概览", "智能预警"):
        advice = position_advice(get_history(symbol, period="daily", start="20230101"))
    else:
        advice = {}

    if page == "概览":
        page_home(symbol, daily_df, pred, pd.DataFrame(), advice)
    elif page == "行情":
        df = load_history(symbol, period, adjust, start)
        if period == "daily":
            df = daily_df
        page_quote(df)
    elif page == "财报":
        page_finance(symbol)
    elif page == "舆情":
        page_news(symbol, use_deep)
    elif page == "AI预测":
        page_predict(symbol)
    elif page == "量化回测":
        page_backtest(symbol, threshold)
    elif page == "智能预警":
        page_alert(symbol)

    st.sidebar.divider()
    st.sidebar.caption("本项目仅用于学术研究与技术演示，不构成投资建议。")


if __name__ == "__main__":
    main()
