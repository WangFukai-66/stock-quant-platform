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
    """队列完成时清预测缓存并请求整页刷新（幂等，返回是否需要 rerun）。"""
    qid = q.get("started_at")
    if st.session_state.get("_cleared_queue") == qid:
        return False
    load_prediction.clear()
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

    pages = ["概览", "行情", "财报", "舆情", "AI预测", "量化回测", "智能预警"]
    if train_available():
        pages.append("后台管理")
    page = st.sidebar.radio("功能页签", pages)

    if page == "后台管理":
        page_admin()
        sidebar_training_widget()
        return

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
    sidebar_training_widget()
    st.sidebar.divider()
    st.sidebar.caption("本项目仅用于学术研究与技术演示，不构成投资建议。")


if __name__ == "__main__":
    main()
