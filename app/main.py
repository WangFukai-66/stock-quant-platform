"""P1-09 Streamlit 六页签看板。

启动: streamlit run app/main.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alert.rules import generate_signals, latest_signals, position_advice
from app.about import page_about
from app.components import STATE_NAMES, kline_with_signals, market_state
from app.style import inject_style, PLOTLY_CONFIG
from backtest.benchmark import buy_hold_equity, excess_metrics, index_equity
from backtest.engine import BacktestEngine, BacktestConfig
from data.fetcher import fetch_index_daily, fetch_news

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
inject_style()
pio.templates.default = "plotly_white"  # 全站图表统一白色模板

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
def load_sentiment(symbol: str) -> float | None:
    """最新舆情情绪均值（规则词典模式），无数据返回 None。"""
    news = fetch_news(symbol)
    if news is None or news.empty:
        return None
    scored, daily = analyze_news(news, use_deep=False)
    if not daily.empty and "情绪均值" in daily.columns:
        return float(daily["情绪均值"].iloc[-1])
    if not scored.empty:
        return float(scored["情绪分"].mean())
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_today_signals(symbol: str, start: str) -> pd.DataFrame:
    """最新一日的全部信号（缓存版，避免每次页面切换重算全量信号）。"""
    return latest_signals(get_history(symbol, period="daily", start=start))


@st.cache_data(ttl=3600, show_spinner=False)
def load_advice(symbol: str) -> dict:
    """仓位建议（基于近三年日线，口径与 position_advice 原调用一致）。"""
    return position_advice(get_history(symbol, period="daily", start="20230101"))


@st.cache_data(ttl=3600, show_spinner=False)
def load_scored_news(symbol: str, use_deep: bool):
    """新闻情绪打分与日度聚合（缓存版；无新闻返回 (None, None)）。"""
    news = load_news(symbol)
    if news is None or news.empty:
        return None, None
    return analyze_news(news, use_deep=use_deep)


@st.cache_data(ttl=86400)
def load_index_equity(start: str = "20150101"):
    """沪深300 归一化净值曲线（接口失败降级为 None）。"""
    return index_equity("sh000300", start=start)


@st.cache_data(ttl=3600)
def load_market_overview() -> list[dict]:
    """上证/深证成指/创业板指/沪深300 最新收盘与涨跌幅（接口失败逐个跳过）。"""
    items = [("sh000001", "上证指数"), ("sz399001", "深证成指"),
             ("sz399006", "创业板指"), ("sh000300", "沪深300")]
    out = []
    for code, name in items:
        try:
            idx = fetch_index_daily(code, start="20250101")
        except Exception:
            continue
        if idx is None or idx.empty or "收盘" not in idx.columns:
            continue
        last = idx.iloc[-1]
        prev = idx.iloc[-2] if len(idx) > 1 else last
        pct = (float(last["收盘"]) / float(prev["收盘"]) - 1) * 100
        out.append({"名称": name, "收盘": float(last["收盘"]), "涨跌幅": round(pct, 2)})
    return out


def market_strip():
    """概览页顶部市场指数条（红涨绿跌，接口不可达时静默隐藏）。"""
    rows = load_market_overview()
    if not rows:
        return
    cols = st.columns(len(rows))
    for col, r in zip(cols, rows):
        col.metric(f"{r['名称']}", f"{r['收盘']:.2f}", f"{r['涨跌幅']:+.2f}%",
                   delta_color="inverse")
    st.caption("市场指数 · akshare 数据源（本地缓存 7 天）")


@st.cache_data(ttl=86400)
def threshold_sensitivity(symbol: str) -> pd.DataFrame:
    """回测买入阈值敏感度（参数快速寻优展示）：0.50~0.70 步长 0.02。"""
    pred = latest_prediction(symbol)
    if pred is None or pred.empty:
        return pd.DataFrame()
    ohlc = get_history(symbol, period="daily", adjust="qfq", start="20150101")
    merged = ohlc.merge(pred[["日期", "预测概率"]], on="日期", how="inner")
    if merged.empty:
        return pd.DataFrame()
    rows = []
    for th in np.arange(0.50, 0.71, 0.02):
        m = BacktestEngine(BacktestConfig(threshold=float(th))).run(
            merged, merged["预测概率"]).metrics
        rows.append({"阈值": round(float(th), 2), "总收益率": m["总收益率"],
                     "夏普比率": m["夏普比率"], "交易次数": m["交易次数"]})
    return pd.DataFrame(rows)


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


# ---------------------------------------------------------------- 后台训练 ----

QUEUE_PATH = PROJECT_ROOT / "results" / "train_queue.json"
SYNC_STATE_PATH = PROJECT_ROOT / "results" / ".sync_state.json"


def venv_train_python() -> str | None:
    """本地训练解释器（.venv-train，云端仓库无此目录）。"""
    p = PROJECT_ROOT / ".venv-train" / "bin" / "python"
    return str(p) if p.exists() else None


def train_available() -> bool:
    """后台训练是否可用（仅本地：有 .venv-train 或当前进程自带 torch）。"""
    if venv_train_python() is not None:
        return True
    try:
        return importlib.util.find_spec("torch") is not None
    except Exception:
        return False


def train_python() -> str | None:
    """训练子进程的解释器（云端为 None）。"""
    if venv_train_python() is not None:
        return venv_train_python()
    try:
        if importlib.util.find_spec("torch") is not None:
            return sys.executable
    except Exception:
        pass
    return None


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def status_path_of(symbol: str) -> Path:
    return PROJECT_ROOT / "results" / f"training_{symbol}.json"


def log_path_of(symbol: str) -> Path:
    return PROJECT_ROOT / "results" / f"train_log_{symbol}.log"


def read_training_status(symbol: str) -> dict:
    """读取单支训练状态；running 但进程已死 → 标记 error（防看板重启后死锁）。"""
    stt = _read_json(status_path_of(symbol)) or {}
    if stt.get("status") == "running" and stt.get("pid"):
        try:
            os.kill(int(stt["pid"]), 0)
        except OSError:
            stt = {**stt, "status": "error", "error": "训练进程意外退出（看板已重启？）"}
            _write_json(status_path_of(symbol), stt)
    return stt


def tail_of(path: Path, n: int = 6) -> str:
    """日志尾部 n 行（文件不存在/读取失败返回空串）。"""
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[-n:])
    except Exception:
        return ""


def read_queue() -> dict | None:
    return _read_json(QUEUE_PATH)


def start_training(symbol: str, quick: bool) -> dict:
    """后台子进程启动单支训练（stdout 重定向到日志文件）。"""
    py = train_python()
    if py is None:
        return {"ok": False, "error": "未检测到本地训练环境（.venv-train）"}
    log = log_path_of(symbol)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            [py, "-m", "scripts.train_all", "--symbol", symbol,
             "--status", str(status_path_of(symbol)),
             *(["--quick"] if quick else [])],
            cwd=str(PROJECT_ROOT), stdout=f, stderr=subprocess.STDOUT)
    return {"ok": True, "pid": proc.pid}


def start_queue(symbols: list[str], quick: bool) -> dict:
    """启动串行训练队列；已有队列在跑时拒绝（全局互斥）。"""
    q = read_queue()
    if q and q.get("status") == "running":
        return {"ok": False, "error": f"已有训练队列在运行（{q['queue'][q['index']]}），请等待完成"}
    queue = {"queue": symbols, "index": 0,
             "mode": "quick" if quick else "standard",
             "status": "running", "failures": {},
             "started_at": time.time()}
    r = start_training(symbols[0], quick)
    if r["ok"]:
        queue["pid"] = r["pid"]
    else:
        queue["status"] = "error"
        queue["error"] = r["error"]
    _write_json(QUEUE_PATH, queue)
    return {"ok": True, "queue": queue}


def _advance_queue(q: dict) -> dict:
    """当前支结束后推进队列：启动下一支或标记整体完成。"""
    symbols, idx = q["queue"], q["index"]
    if idx + 1 >= len(symbols):
        q["status"] = "done"
        q["finished_at"] = time.time()
        return q
    nxt = symbols[idx + 1]
    r = start_training(nxt, q["mode"] == "quick")
    if not r["ok"]:
        q["status"] = "error"
        q["error"] = f"无法启动 {nxt} 训练：{r['error']}"
        return q
    q["index"] = idx + 1
    q["pid"] = r["pid"]
    return q


def drive_queue() -> dict | None:
    """队列调度（在进度 fragment 中周期调用）：收尾已完成支、启动下一支。"""
    q = read_queue()
    if not q or q.get("status") != "running":
        return q
    while True:
        idx = q["index"]
        if idx >= len(q["queue"]):
            q["status"] = "done"
            q["finished_at"] = time.time()
            break
        cur = q["queue"][idx]
        s = read_training_status(cur).get("status")
        if s == "running" or s is None:
            break                      # 当前支仍在训练（或状态未落盘），等待
        if s == "error":
            q["failures"][cur] = read_training_status(cur).get("error") or "训练失败（详见日志）"
        q = _advance_queue(q)
        if q["status"] != "running":
            break
    _write_json(QUEUE_PATH, q)
    return q


def on_queue_done(q: dict) -> bool:
    """队列完成时清预测及派生缓存并请求整页刷新（幂等，返回是否需要 rerun）。"""
    qid = q.get("started_at")
    if st.session_state.get("_cleared_queue") == qid:
        return False
    load_prediction.clear()
    load_backtest.clear()
    load_today_signals.clear()
    load_advice.clear()
    threshold_sensitivity.clear()
    st.session_state["_cleared_queue"] = qid
    return True


def scan_trained_symbols() -> list[str]:
    """扫描已训练股票（带后缀预测文件；无后缀旧文件视为默认标的）。"""
    symbols: list[str] = []
    results = PROJECT_ROOT / "results"
    for p in sorted(results.glob("fusion_predictions_*.csv")):
        m = re.fullmatch(r"fusion_predictions_(\d{6})\.csv", p.name)
        if m and m.group(1) not in symbols:
            symbols.append(m.group(1))
    if (results / "fusion_predictions.csv").exists() and DEFAULT_SYMBOL not in symbols:
        symbols.append(DEFAULT_SYMBOL)
    return symbols


def _max_product_mtime(symbol: str) -> float:
    """该股票预测/指标产物的最新修改时间（判断是否待同步）。"""
    newest = 0.0
    results = PROJECT_ROOT / "results"
    for name in ("fusion", "xgb", "lstm", "transformer"):
        for kind in ("predictions", "metrics"):
            ext = ".csv" if kind == "predictions" else ".json"
            p = results / f"{name}_{kind}_{symbol}{ext}"
            try:
                newest = max(newest, p.stat().st_mtime)
            except OSError:
                pass
    return newest


def load_sync_state() -> dict:
    return _read_json(SYNC_STATE_PATH, {}) or {}


def pending_sync_symbols(trained: list[str]) -> list[str]:
    """待同步 = 已训练且产物比上次同步更新（或从未同步）。"""
    state = load_sync_state()
    return [s for s in trained
            if s not in state or _max_product_mtime(s) > float(state[s].get("mtime", 0))]


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
              signals: pd.DataFrame, advice: dict, start: str):
    st.header("概览")
    market_strip()
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    chg = (last["收盘"] / prev["收盘"] - 1) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最新收盘价", f"{last['收盘']:.2f}", f"{chg:+.2f}%", delta_color="inverse")
    c2.metric("最新成交量", f"{last['成交量']/1e4:.1f} 万手")
    if pred is not None and not pred.empty:
        p_last = float(pred.iloc[-1]["预测概率"])
        c3.metric("次日上涨概率", f"{p_last*100:.1f}%",
                  "看涨" if p_last >= 0.55 else ("看跌" if p_last < 0.45 else "中性"))
    else:
        c3.metric("次日上涨概率", "未训练")
    today_sig = len(load_today_signals(symbol, start))
    c4.metric("今日预警信号", f"{today_sig} 条")

    # 风险与情绪监控指标卡
    d1, d2, d3, d4 = st.columns(4)
    senti = load_sentiment(symbol)
    if senti is not None:
        d1.metric("舆情情绪均值", f"{senti:+.2f}", "正面" if senti > 0.05 else ("负面" if senti < -0.05 else "中性"))
    else:
        d1.metric("舆情情绪均值", "暂无数据")
    ret = df["收盘"].pct_change()
    vol20 = float(ret.tail(20).std() * np.sqrt(252) * 100)
    d2.metric("年化波动率(20日)", f"{vol20:.1f}%")
    if "ATR" in df.columns and pd.notna(last["ATR"]):
        d3.metric("ATR(14)", f"{last['ATR']:.2f}", f"占收盘 {last['ATR']/last['收盘']*100:.1f}%")
    else:
        d3.metric("ATR(14)", "--")
    high20 = float(df["收盘"].tail(20).max())
    dd20 = (high20 - last["收盘"]) / high20 * 100
    d4.metric("距20日高点回撤", f"{dd20:.1f}%")

    if pred is None or pred.empty:
        st.caption(trained_hint(symbol))

    st.subheader("仓位建议")
    st.info(f"**{advice['建议仓位']}** — {advice['依据']}（买入 {advice.get('买入信号', 0)} / "
            f"卖出 {advice.get('卖出信号', 0)} / 风险 {advice.get('风险信号', 0)}）")

    st.subheader("近 120 日走势与信号")
    st.plotly_chart(kline_with_signals(df, signals, days=120, title=f"{symbol} 近120日"),
                    width='stretch', config=PLOTLY_CONFIG)


def page_quote(df: pd.DataFrame):
    st.header("行情看板")
    st.plotly_chart(kline_fig(df), width='stretch', config=PLOTLY_CONFIG)
    with st.expander("原始数据（近 30 行）"):
        st.dataframe(df.tail(30), width='stretch')
    _vision_section(df)


@st.cache_data(ttl=3600, show_spinner=False)
def load_pattern_result(df: pd.DataFrame) -> dict:
    """K线形态识别结果（近 60 日渲染 + CNN 概率），整包缓存避免每次切换重算。

    返回 status: no_deps(视觉依赖缺失) / no_model(权重未训练) /
    no_torch(推理依赖缺失) / error(推理失败) / ok(含 img 与 probs)。
    """
    try:
        from vision.render import PATTERNS, render_candlestick
    except Exception as exc:  # 云端无 matplotlib/PIL 依赖
        return {"status": "no_deps", "detail": type(exc).__name__}
    img = render_candlestick(df.tail(60))
    weight_path = PROJECT_ROOT / "models" / "pattern_cnn.pt"
    if not weight_path.exists():
        return {"status": "no_model", "img": img}
    try:
        from vision.cnn_pattern import load_pattern_cnn, predict_pattern_probs
    except Exception as exc:  # 云端禁用 torch
        return {"status": "no_torch", "detail": type(exc).__name__}
    try:
        model = load_pattern_cnn(weight_path)
        probs = predict_pattern_probs(model, [img])[0]
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
    return {"status": "ok", "img": img, "patterns": list(PATTERNS),
            "probs": [float(p) for p in probs]}


def _vision_section(df: pd.DataFrame):
    """K线形态 CNN 识别（第 4 路融合信号）：依赖/权重缺失时优雅降级提示。"""
    with st.expander("K线形态识别（CNN）", expanded=False):
        st.caption("ResNet18 微调识别 5 类 K 线形态（头肩顶/双底/上升三角/箱体/突破），"
                   "作为融合引擎的第 4 路信号（合成数据训练，结论仅供研究参考）")
        res = load_pattern_result(df)
        status = res["status"]
        if status == "no_deps":
            st.info(f"当前环境未安装视觉渲染依赖（{res['detail']}），跳过形态识别展示。")
            return
        if status == "no_model":
            col1, col2 = st.columns([1, 2])
            col1.image(res["img"], caption="近60日标准化K线渲染", width=160)
            col2.warning("形态识别模型未训练。本地训练命令：`python -m vision.cnn_pattern`")
            return
        if status == "no_torch":
            st.info(f"当前环境未安装 torch 推理依赖（{res['detail']}），无法加载形态模型。")
            return
        if status == "error":
            st.warning(f"形态识别推理失败：{res['detail']}")
            return
        top = sorted(zip(res["patterns"], res["probs"]), key=lambda x: -x[1])
        col1, col2 = st.columns([1, 2])
        col1.image(res["img"], caption="近60日标准化K线渲染", width=160)
        fig = go.Figure(go.Bar(x=[t for t, _ in top], y=[p * 100 for _, p in top],
                               marker_color="#3b82f6"))
        fig.update_layout(height=280, title="形态概率 Top5", yaxis_title="概率%")
        col2.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)


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
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)


def page_news(symbol: str, use_deep: bool):
    st.header("舆情监控")
    scored, daily = load_scored_news(symbol, use_deep)
    if scored is None or daily is None:
        st.warning("暂无新闻数据")
        return
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
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)


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
    st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
    st.dataframe(pred.tail(30), width='stretch')
    st.download_button("下载预测数据 CSV", pred.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"prediction_{symbol}.csv", key=f"dl_pred_{symbol}")

    # 各模型指标（XGB / LSTM / Transformer / 融合）
    fusion_json = find_fusion_metrics(symbol)
    st.subheader("模型指标")
    cols = st.columns(4)
    for i, name in enumerate(("xgb", "lstm", "transformer", "fusion")):
        if name == "fusion":
            if fusion_json:
                s = fusion_json.get("stacking") or {}
                d = fusion_json.get("dynamic") or {}
                cols[i].markdown(f"**FUSION**  \nstacking acc={s.get('accuracy')}  \n"
                                 f"dynamic acc={d.get('accuracy')}")
            else:
                cols[i].markdown(f"**FUSION**  \n未训练")
            continue
        m = find_metrics(symbol, name) or metrics_from_pred(symbol, name)
        if m:
            auc = m.get("auc")
            cols[i].markdown(f"**{name.upper()}**  \naccuracy={m.get('accuracy')}  \n"
                             f"auc={auc if auc is not None else '--'}")
        else:
            cols[i].markdown(f"**{name.upper()}**  \n未训练")

    # 模型对比（融合 vs 单模型，来自 fusion_metrics.json 的对比表）
    st.subheader("模型对比（Stacking / 动态权重 vs 单模型）")
    if fusion_json and fusion_json.get("compare"):
        comp = pd.DataFrame(fusion_json["compare"])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=comp["模型"], y=comp["accuracy"], name="accuracy",
                             marker_color="#3b82f6"), secondary_y=False)
        fig.add_trace(go.Scatter(x=comp["模型"], y=comp["auc"], name="auc",
                                 mode="lines+markers", line=dict(color="#ef4444")), secondary_y=True)
        fig.update_layout(height=350, yaxis_title="accuracy", yaxis2_title="auc")
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
    else:
        st.caption("暂无融合对比数据（本地训练后生成 fusion_metrics.json）")

    # 动态权重机制可视化：市场状态四象限 + 元学习器权重
    st.subheader("动态权重机制（市场状态自适应）")
    c1, c2, c3 = st.columns(3)
    if "收盘" in pred.columns:
        states = market_state(pred["收盘"])
        names = [STATE_NAMES[s] for s in sorted(states.unique())]
        counts = [int((states == s).sum()) for s in sorted(states.unique())]
        pie = go.Figure(go.Pie(labels=names, values=counts, hole=0.45))
        pie.update_layout(height=300, title="测试期市场状态分布")
        c1.plotly_chart(pie, width='stretch', config=PLOTLY_CONFIG)
    else:
        c1.caption("预测文件无收盘序列，无法计算市场状态")
    if fusion_json and fusion_json.get("weights"):
        w = fusion_json["weights"]
        bar = go.Figure(go.Bar(x=list(w.keys()), y=[v * 100 for v in w.values()],
                               marker_color=["#3b82f6", "#8b5cf6", "#f59e0b"]))
        bar.update_layout(height=300, title="Stacking 元学习器权重", yaxis_title="权重%")
        c2.plotly_chart(bar, width='stretch', config=PLOTLY_CONFIG)
    else:
        c2.caption("暂无元学习器权重数据")
    c3.markdown(
        "**机制说明**\n\n"
        "- 市场状态四象限：20日波动率分位（高/低）× 趋势方向（趋势/震荡）\n\n"
        "- 每个象限在验证集上独立学习一组融合权重，测试期按当日状态自适应选用\n\n"
        "- 样本不足 20 的象限退化为等权平均，避免过拟合\n\n"
        "- 对比上方条形图：Stacking / 动态权重与单模型在测试集上的 accuracy 与 AUC"
    )


def find_fusion_metrics(symbol: str) -> dict | None:
    """读取融合对比结果 fusion_metrics.json（含 compare 表与元学习器权重）。"""
    for suffix in (f"_{symbol}", "" if symbol == DEFAULT_SYMBOL else None):
        if suffix is None:
            continue
        p = PROJECT_ROOT / "results" / f"fusion_metrics{suffix}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def metrics_from_pred(symbol: str, name: str) -> dict | None:
    """从预测文件即时计算 accuracy/auc（train_all 仅落盘融合指标）。

    云端无 sklearn 依赖时降级为仅 accuracy。
    """
    pred = load_prediction(str(PROJECT_ROOT / "results" / f"{name}_predictions_{symbol}.csv"))
    if pred is None or pred.empty or "真实标签" not in pred.columns:
        return None
    y = pred["真实标签"].astype(int)
    p = pred["预测概率"].astype(float)
    acc = round(float(((p >= 0.5).astype(int) == y).mean()), 4)
    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        auc = round(float(roc_auc_score(y, p)), 4)
    except Exception:
        pass
    return {"accuracy": acc, "auc": auc}


@st.cache_data(ttl=3600, show_spinner=False)
def load_backtest(symbol: str, threshold: float):
    """回测 + 基准对比（引擎/买入持有/沪深300/超额指标）整包缓存。

    返回 {"error": "no_pred"} / {"error": "no_overlap"} 或完整结果包。
    """
    pred = latest_prediction(symbol)
    if pred is None:
        return {"error": "no_pred"}
    ohlc = get_history(symbol, period="daily", adjust="qfq", start="20150101")
    merged = ohlc.merge(pred[["日期", "预测概率"]], on="日期", how="inner")
    if merged.empty:
        return {"error": "no_overlap"}
    cfg = BacktestConfig(threshold=threshold)
    result = BacktestEngine(cfg).run(merged, merged["预测概率"])
    bh = buy_hold_equity(merged, cfg.initial_cash)
    idx = load_index_equity()
    return {
        "metrics": result.metrics,
        "equity": result.equity,
        "drawdown": result.drawdown,
        "trades": result.trades,
        "bh": bh,
        "idx": idx,
        "ex_bh": excess_metrics(result.equity, bh),
        "ex_idx": excess_metrics(result.equity, idx) if idx is not None else {},
    }


def page_backtest(symbol: str, threshold: float):
    st.header("量化回测")
    st.caption("严格模拟 A 股规则：T+1、涨跌停不可成交、佣金万2.5、印花税卖出千1、滑点1分；"
               "基准对比：标的买入持有 + 沪深300 指数")
    bundle = load_backtest(symbol, threshold)
    if bundle.get("error") == "no_pred":
        st.warning(trained_hint(symbol))
        return
    if bundle.get("error") == "no_overlap":
        st.warning("预测与行情日期无交集")
        return

    m = bundle["metrics"]
    equity, drawdown = bundle["equity"], bundle["drawdown"]
    bh, idx = bundle["bh"], bundle["idx"]
    ex_bh, ex_idx = bundle["ex_bh"], bundle["ex_idx"]
    trades = bundle["trades"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总收益率", f"{m['总收益率']}%")
    c2.metric("年化收益率", f"{m['年化收益率']}%")
    c3.metric("夏普比率", f"{m['夏普比率']}")
    c4.metric("最大回撤", f"{m['最大回撤']}%")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("胜率(日)", f"{m['胜率(日)']}%", f"盈利/亏损日 {m['盈利日/亏损日']}")
    c6.metric("交易次数", m["交易次数"])
    c7.metric("超额收益(买入持有)", f"{ex_bh.get('超额收益', '--')}%")
    c8.metric("超额收益(沪深300)", f"{ex_idx.get('超额收益', '--')}%"
             if idx is not None else "指数不可达")
    st.caption(f"回测区间：{m['回测区间']}")

    # 资金曲线（策略 vs 基准，归一化）
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05,
                        subplot_titles=("策略 vs 基准（归一化净值）", "回撤"))
    fig.add_trace(go.Scatter(x=equity.index,
                             y=equity.values / equity.values[0],
                             name="策略", line=dict(color="#2563eb", width=2)), row=1, col=1)
    if not bh.empty:
        fig.add_trace(go.Scatter(x=bh.index, y=bh.values / bh.values[0],
                                 name="买入持有", line=dict(color="#9ca3af", width=1)), row=1, col=1)
    if idx is not None and not idx.empty:
        fig.add_trace(go.Scatter(x=idx.index, y=idx.values,
                                 name="沪深300", line=dict(color="#f59e0b", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values * 100,
                             name="回撤%", line=dict(color="#dc2626"), fill="tozeroy"), row=2, col=1)
    fig.update_layout(height=550, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    if trades is not None and not trades.empty:
        st.subheader("交易明细")
        st.caption("收益率未计手续费与滑点，仅供展示参考")
        trades_full = _enrich_trades(trades)
        st.dataframe(trades_full.tail(30), width='stretch')
        st.download_button("下载交易明细 CSV", trades_full.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"trades_{symbol}.csv", key=f"dl_trades_{symbol}")

    # 参数寻优展示：买入阈值敏感度
    st.subheader("参数寻优（买入阈值敏感度）")
    sens = threshold_sensitivity(symbol)
    if not sens.empty:
        best = sens.loc[sens["夏普比率"].idxmax()]
        st.caption(f"夏普最优阈值 {best['阈值']}（夏普 {best['夏普比率']}，收益 {best['总收益率']}%）；"
                   "完整贝叶斯寻优见 scripts/optimize.py（optuna）")
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        fig2.add_trace(go.Scatter(x=sens["阈值"], y=sens["总收益率"], name="总收益率%",
                                  mode="lines+markers", line=dict(color="#2563eb")),
                       secondary_y=False)
        fig2.add_trace(go.Scatter(x=sens["阈值"], y=sens["夏普比率"], name="夏普比率",
                                  mode="lines+markers", line=dict(color="#dc2626")),
                       secondary_y=True)
        fig2.update_layout(height=320, xaxis_title="买入概率阈值",
                           yaxis_title="总收益率%", yaxis2_title="夏普比率")
        st.plotly_chart(fig2, width='stretch', config=PLOTLY_CONFIG)


def _enrich_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """交易明细增强：按时间顺序买卖配对，补充单笔收益率与持仓天数。"""
    if trades is None or trades.empty:
        return trades
    out = trades.copy()
    out["收益率%"] = ""
    out["持仓天数"] = ""
    open_buy = None
    for idx, row in out.iterrows():
        if row["方向"] == "买入":
            open_buy = (row["日期"], row["价格"], row["股数"])
        elif row["方向"] == "卖出" and open_buy is not None:
            bd, bp, _ = open_buy
            out.at[idx, "收益率%"] = round((row["价格"] - bp) / bp * 100, 2)
            out.at[idx, "持仓天数"] = (pd.to_datetime(row["日期"]) - pd.to_datetime(bd)).days
            open_buy = None
    return out


def page_alert(symbol: str, daily_df: pd.DataFrame, start: str):
    st.header("智能预警")
    st.caption("规则引擎基于当日收盘数据触发信号（次日可执行）；演示版在页面内模拟推送，不接任何实盘通道")
    sig = load_signals(symbol, "20230101")
    advice = load_advice(symbol)

    st.subheader("当前仓位建议")
    c1, c2 = st.columns([3, 1])
    c1.info(f"**{advice['建议仓位']}** — {advice['依据']}")
    if c2.button("模拟推送预警", width='stretch'):
        st.toast(f"【{symbol}】仓位建议：{advice['建议仓位']}（{advice['依据']}）", icon="🔔")

    # 风险监控面板
    last = daily_df.iloc[-1]
    ret = daily_df["收盘"].pct_change()
    vol20 = float(ret.tail(20).std() * np.sqrt(252) * 100)
    vol_hist = ret.rolling(20).std() * np.sqrt(252) * 100
    vol_pct = float((vol_hist.dropna() <= vol20).mean() * 100)
    high20 = float(daily_df["收盘"].tail(20).max())
    dd20 = (high20 - last["收盘"]) / high20 * 100
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("年化波动率(20日)", f"{vol20:.1f}%", f"历史分位 {vol_pct:.0f}%",
              delta_color="inverse" if vol_pct > 70 else "normal")
    r2.metric("ATR(14)", f"{last['ATR']:.2f}" if "ATR" in daily_df.columns else "--",
              f"占收盘 {last['ATR']/last['收盘']*100:.1f}%" if "ATR" in daily_df.columns and pd.notna(last['ATR']) else None)
    r3.metric("距20日高点回撤", f"{dd20:.1f}%",
              "超10%触发回撤预警" if dd20 >= 10 else None,
              delta_color="inverse")
    r4.metric("今日信号", f"{len(load_today_signals(symbol, start))} 条")

    if sig is None or sig.empty:
        st.info("近一年无触发信号")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("累计信号", len(sig))
        c2.metric("买入信号", int((sig["方向"] == "买入").sum()))
        c3.metric("卖出信号", int((sig["方向"] == "卖出").sum()))
        c4.metric("风险信号", int((sig["方向"] == "风险").sum()))

        st.subheader("近 120 日信号标注")
        st.plotly_chart(kline_with_signals(daily_df, sig, days=120,
                                           title=f"{symbol} K线与预警信号"),
                        width='stretch', config=PLOTLY_CONFIG)

        st.subheader("最近信号")
        st.dataframe(sig.tail(30), width='stretch')
        st.download_button("下载全部信号 CSV", sig.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"signals_{symbol}.csv", key=f"dl_sig_{symbol}")

    with st.expander("预警规则说明"):
        st.markdown(
            "| 信号 | 方向 | 触发条件 | 操作建议 |\n"
            "|---|---|---|---|\n"
            "| 金叉 | 买入 | MA5 上穿 MA20 | 可建仓 |\n"
            "| 死叉 | 卖出 | MA5 下穿 MA20 | 可减仓 |\n"
            "| 涨停触板 | 风险 | 当日涨幅 ≥ 9.9% | 不追高 |\n"
            "| 跌停触板 | 风险 | 当日跌幅 ≤ -9.9% | 不抄底 |\n"
            "| 放量异动 | 关注 | 量比 > 2 | 重点观察 |\n"
            "| MACD底背离 | 买入 | 价创20日新低而 DIF 未同步 | 关注反弹 |\n"
            "| MACD顶背离 | 卖出 | 价创20日新高而 DIF 未同步 | 警惕回调 |\n"
            "| 回撤预警 | 风险 | 自20日高点回撤 ≥ 10% | 控制仓位 |\n\n"
            "信号在当日收盘后基于当日数据触发（次日可执行），严格避免未来函数。"
        )


def page_admin():
    st.header("后台管理（仅本地环境显示）")
    st.caption("训练在后台执行，可切换到其他页签使用；云端用户看不到本页签。")

    # ---- 已训练股票状态表 ----
    st.subheader("已训练股票")
    trained = scan_trained_symbols()
    state = load_sync_state()
    pending = pending_sync_symbols(trained)
    if not trained:
        st.info("暂无已训练股票（默认标的 600519 的无后缀旧版预测也算已训练）")
    else:
        rows = [{"股票代码": s,
                 "同步状态": "已同步云端" if s not in pending else "仅本地",
                 "上次同步": state.get(s, {}).get("time", "-")}
                for s in sorted(trained)]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ---- 批量训练 ----
    st.subheader("批量训练")
    admin_training_fragment()
    q = read_queue()
    if not q or q.get("status") != "running":
        codes = st.text_area("输入股票代码（6 位数字，逗号/空格/换行分隔）",
                             placeholder="例如：000001, 600036, 601318")
        mode = st.radio("训练模式", ["快速（约 3~6 分钟/支）", "标准（约 10~15 分钟/支）"],
                        horizontal=True)
        if st.button("开始训练", type="primary"):
            symbols = list(dict.fromkeys(re.findall(r"\d{6}", codes)))
            if not symbols:
                st.error("未识别到有效的 6 位股票代码")
            else:
                r = start_queue(symbols, quick=mode.startswith("快速"))
                if r["ok"]:
                    st.success(f"已启动训练队列：{', '.join(symbols)}")
                    st.rerun()
                else:
                    st.error(r["error"])

    st.divider()

    # ---- 同步云端 ----
    st.subheader("同步云端")
    trained = scan_trained_symbols()
    pending = pending_sync_symbols(trained)
    if not pending:
        st.success("所有已训练股票均已同步云端（云端重建需几分钟后可见）")
    else:
        st.info("待同步股票：" + ", ".join(sorted(pending)))
        confirm = st.checkbox("我已确认：仅提交上述股票的预测/指标文件到 cloud 分支，不触碰其他文件")
        if st.button("一键同步云端", type="primary"):
            if not confirm:
                st.error("请先勾选确认")
            else:
                with st.spinner("执行安全同步中（add → commit → push）..."):
                    r = subprocess.run(
                        [sys.executable, "-m", "scripts.sync_cloud", ",".join(pending)],
                        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300)
                try:
                    res = json.loads(r.stdout.strip().splitlines()[-1])
                except Exception:
                    res = {"ok": False, "error": r.stdout.strip() or r.stderr.strip()}
                if res.get("ok"):
                    st.success(f"同步成功（commit {res.get('commit')}）。"
                               f"Streamlit Cloud 自动重建中，几分钟后云端可见。")
                else:
                    st.error("同步失败：" + str(res.get("error")))


# ---------------------------------------------------------------- 侧边栏进度 ----

def _sidebar_training_widget_body():
    """侧边栏底部训练进度（局部自动刷新，不影响其他页签功能）。"""
    q = drive_queue()
    if not q:
        return
    if q.get("status") == "running":
        symbols = q["queue"]
        idx = q["index"]
        cur = symbols[idx]
        stt = read_training_status(cur)
        st.sidebar.progress(int(stt.get("progress", 0)) / 100,
                            text=f"训练 {cur}（{idx + 1}/{len(symbols)}）")
        log = tail_of(log_path_of(cur), 3)
        if log:
            st.sidebar.caption(log)
    elif q.get("status") == "done":
        if on_queue_done(q):
            st.rerun()


def _admin_training_fragment_body():
    """后台管理页训练进度区（局部自动刷新）。"""
    q = drive_queue()
    if not q:
        return
    if q.get("status") == "running":
        symbols = q["queue"]
        idx = q["index"]
        cur = symbols[idx]
        stt = read_training_status(cur)
        st.progress(int(stt.get("progress", 0)) / 100,
                    text=f"正在训练 {cur}（第 {idx + 1}/{len(symbols)} 支，{q['mode']} 模式）")
        log = tail_of(log_path_of(cur), 8)
        if log:
            st.code(log)
        failures = q.get("failures") or {}
        if failures:
            st.warning("已失败：" + "；".join(f"{k}: {v}" for k, v in failures.items()))
    elif q.get("status") == "done":
        failures = q.get("failures") or {}
        if failures:
            st.warning("队列完成，部分失败：" + "；".join(f"{k}: {v}" for k, v in failures.items()))
        else:
            st.success(f"队列训练完成：{', '.join(q['queue'])}")
    elif q.get("status") == "error":
        st.error(q.get("error") or "训练队列执行出错")


# 旧版 Streamlit 无 fragment 时降级为普通渲染（进度靠手动刷新）
if hasattr(st, "fragment"):
    sidebar_training_widget = st.fragment(run_every=3)(_sidebar_training_widget_body)
    admin_training_fragment = st.fragment(run_every=3)(_admin_training_fragment_body)
else:
    def sidebar_training_widget():
        _sidebar_training_widget_body()

    def admin_training_fragment():
        _admin_training_fragment_body()


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
    elif symbol == DEFAULT_SYMBOL:
        st.sidebar.caption(f"**{DEFAULT_NAME}**")
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

    pages = ["项目介绍", "概览", "行情", "财报", "舆情", "AI预测", "量化回测", "智能预警"]
    if train_available():
        pages.append("后台管理")
    page = st.sidebar.radio("功能页签", pages)

    st.sidebar.divider()
    with st.sidebar.expander("关于本项目"):
        st.caption("雏雁计划项目 | 国际学院 · 智能科学与技术专业")
        st.caption("线上演示：https://stock-quant-platform.streamlit.app/")
        st.caption("开源仓库：WangFukai-66/stock-quant-platform")
    st.sidebar.caption("本项目仅用于学术研究与技术演示，不构成投资建议。")

    if page == "后台管理":
        page_admin()
        sidebar_training_widget()
        return

    sidebar_training_widget()

    if page == "项目介绍":
        page_about()
        return

    with st.spinner("加载数据中..."):
        daily_df = load_indicators(symbol, start)
        if daily_df is None or daily_df.empty:
            st.error(f"未获取到 {symbol} 的行情数据，请检查股票代码")
            return

    pred = latest_prediction(symbol)
    if page in ("概览", "智能预警"):
        advice = load_advice(symbol)
    else:
        advice = {}

    if page == "概览":
        page_home(symbol, daily_df, pred, load_signals(symbol, "20230101"), advice, start)
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
        page_alert(symbol, daily_df, start)


if __name__ == "__main__":
    main()
