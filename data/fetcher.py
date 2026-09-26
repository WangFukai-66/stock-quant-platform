"""P0-02 数据采集模块：akshare 封装 + parquet 本地缓存 + 限速重试。

统一中文列名：日期/开盘/收盘/最高/最低/成交量/成交额/振幅/涨跌幅/涨跌额/换手率
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# 网络兼容性：部分网络环境 IPv6 路由不稳定（东财/新浪接口连接被重置），
# 统一让 socket 解析优先返回 IPv4，保证 akshare 请求稳定（对云端部署同样无害）。
# 同时绕过系统代理直连：东财/新浪均为国内接口，直连最稳定（仅影响本进程）。
def _prefer_ipv4() -> None:
    import os
    import socket as _socket
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")
    _orig = _socket.getaddrinfo

    def _patched(host, *args, **kwargs):
        infos = _orig(host, *args, **kwargs)
        return sorted(infos, key=lambda x: x[0] != _socket.AF_INET)

    _socket.getaddrinfo = _patched


_prefer_ipv4()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_TTL = timedelta(days=7)          # 缓存有效期
MIN_INTERVAL = 1.0                     # 请求最小间隔（秒）
MAX_RETRY = 3                          # 重试次数

DAILY_COLS = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]
MINUTE_COLS = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额"]

_last_request = 0.0


def _rate_limit() -> None:
    """保证两次请求间隔不小于 MIN_INTERVAL 秒。"""
    global _last_request
    elapsed = time.time() - _last_request
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_request = time.time()


def _retry(func, *args, retries: int = MAX_RETRY, **kwargs):
    """带限速与重试的调用包装。"""
    for attempt in range(1, retries + 1):
        try:
            _rate_limit()
            return func(*args, **kwargs)
        except Exception as exc:  # akshare 接口偶发超时/连接重置
            logger.warning("第 %d 次请求失败(%s): %s", attempt, func.__name__, exc)
            if attempt == retries:
                raise
            time.sleep(attempt * 2)


def _with_fallback(primary, backup, primary_name: str = "东财", backup_name: str = "备用"):
    """主数据源失败时降级备用数据源（部分网络环境东财行情接口不可达）。"""
    try:
        return primary()
    except Exception as exc:
        logger.warning("%s接口失败(%s)，降级%s数据源", primary_name, exc, backup_name)
        return backup()


def _market_prefix(symbol: str) -> str:
    """纯数字代码 → 带市场前缀（腾讯/新浪接口要求）：6xx→sh，0/3xx→sz，4/8xx→bj。"""
    if symbol.startswith(("sh", "sz", "bj")):
        return symbol
    if symbol.startswith(("6", "5", "9")):
        return "sh" + symbol
    if symbol.startswith(("4", "8")):
        return "bj" + symbol
    return "sz" + symbol


def _cache_path(kind: str, symbol: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{kind}_{symbol}.parquet"


def _read_cache(path: Path) -> pd.DataFrame | None:
    if path.exists():
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime < CACHE_TTL:
            df = pd.read_parquet(path)
            logger.info("命中缓存 %s (%d 行)", path.name, len(df))
            return df
    return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    if df is not None and not df.empty:
        df.to_parquet(path, index=False)
        logger.info("已缓存 %s (%d 行)", path.name, len(df))


# ---------------------------------------------------------------- 行情 ----

def _fetch_daily_em(symbol: str, adjust: str, start: str, end: str) -> pd.DataFrame:
    """东财日K（主数据源，列最全）。"""
    import akshare as ak
    raw = _retry(ak.stock_zh_a_hist, symbol=symbol, period="daily",
                 start_date=start, end_date=end, adjust=adjust, retries=1)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=DAILY_COLS)
    rename = {"日期": "日期", "开盘": "开盘", "收盘": "收盘", "最高": "最高", "最低": "最低",
              "成交量": "成交量", "成交额": "成交额", "振幅": "振幅", "涨跌幅": "涨跌幅",
              "涨跌额": "涨跌额", "换手率": "换手率"}
    df = raw.rename(columns=rename)
    return df[[c for c in DAILY_COLS if c in df.columns]].copy()


def _fetch_daily_tx(symbol: str, adjust: str, start: str, end: str) -> pd.DataFrame:
    """腾讯日K（备用数据源）：英文列转中文，量/换手率单位对齐东财，补算涨跌幅/振幅。"""
    import akshare as ak
    raw = _retry(ak.stock_zh_a_hist_tx, symbol=_market_prefix(symbol),
                 start_date=start, end_date=end, adjust=adjust)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=DAILY_COLS)
    df = raw.rename(columns={"date": "日期", "open": "开盘", "close": "收盘",
                             "high": "最高", "low": "最低", "volume": "成交量",
                             "amount": "成交额", "turnover": "换手率"})
    df["成交量"] = pd.to_numeric(df["成交量"], errors="coerce") / 100        # 股 → 手
    if "换手率" in df.columns:
        df["换手率"] = pd.to_numeric(df["换手率"], errors="coerce") * 100    # 小数 → %
    close = pd.to_numeric(df["收盘"], errors="coerce")
    df["涨跌幅"] = (close.pct_change() * 100).round(2)
    df["涨跌额"] = close.diff().round(2)
    prev = close.shift(1).replace(0, np.nan)
    df["振幅"] = ((pd.to_numeric(df["最高"], errors="coerce")
                   - pd.to_numeric(df["最低"], errors="coerce")) / prev * 100).round(2)
    return df[[c for c in DAILY_COLS if c in df.columns]].copy()


def fetch_daily(symbol: str, adjust: str = "qfq", start: str = "20150101", end: str | None = None) -> pd.DataFrame:
    """A股日K线（东财主源，失败自动降级腾讯），adjust: qfq 前复权 / hfq 后复权 / "" 不复权。"""
    path = _cache_path(f"daily_{adjust}", symbol)
    cached = _read_cache(path)
    if cached is not None:
        return cached

    end = end or datetime.now().strftime("%Y%m%d")
    df = _with_fallback(
        lambda: _fetch_daily_em(symbol, adjust, start, end),
        lambda: _fetch_daily_tx(symbol, adjust, start, end),
        backup_name="腾讯")
    if df is None or df.empty:
        logger.warning("日K为空: %s", symbol)
        return pd.DataFrame(columns=DAILY_COLS)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    for col in df.columns:
        if col != "日期":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    _write_cache(path, df)
    return df


def _fetch_minute_em(symbol: str, period: str, adjust: str) -> pd.DataFrame:
    """东财分钟线（主数据源）。"""
    import akshare as ak
    raw = _retry(ak.stock_zh_a_hist_min_em, symbol=symbol, period=period, adjust=adjust, retries=1)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=MINUTE_COLS)
    rename = {"时间": "日期", "开盘": "开盘", "收盘": "收盘", "最高": "最高",
              "最低": "最低", "成交量": "成交量", "成交额": "成交额"}
    df = raw.rename(columns=rename)
    return df[[c for c in MINUTE_COLS if c in df.columns]].copy()


def _fetch_minute_tx(symbol: str, period: str, adjust: str) -> pd.DataFrame:
    """腾讯分钟线（备用数据源）。"""
    import akshare as ak
    raw = _retry(ak.stock_zh_a_minute, symbol=_market_prefix(symbol), period=period, adjust=adjust)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=MINUTE_COLS)
    df = raw.rename(columns={"day": "日期", "open": "开盘", "close": "收盘",
                             "high": "最高", "low": "最低", "volume": "成交量"})
    return df[[c for c in MINUTE_COLS if c in df.columns]].copy()


def fetch_minute(symbol: str, period: str = "5", adjust: str = "") -> pd.DataFrame:
    """A股分钟线（东财主源，失败自动降级腾讯）。period: 1/5/15/30/60，默认近 5 个交易日。"""
    path = _cache_path(f"minute_{period}", symbol)
    cached = _read_cache(path)
    if cached is not None:
        return cached

    df = _with_fallback(
        lambda: _fetch_minute_em(symbol, period, adjust),
        lambda: _fetch_minute_tx(symbol, period, adjust),
        backup_name="腾讯")
    if df is None or df.empty:
        logger.warning("分钟线为空: %s", symbol)
        return pd.DataFrame(columns=MINUTE_COLS)
    df["日期"] = pd.to_datetime(df["日期"])
    for col in ["开盘", "收盘", "最高", "最低", "成交量", "成交额"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    _write_cache(path, df)
    return df


def fetch_index_daily(index_code: str = "sh000300", start: str = "20150101") -> pd.DataFrame:
    """指数日K（用于基准对比），如 sh000300 沪深300。"""
    path = _cache_path("index", index_code)
    cached = _read_cache(path)
    if cached is not None:
        return cached
    import akshare as ak
    raw = _retry(ak.stock_zh_index_daily, symbol=index_code)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=DAILY_COLS)
    df = raw.rename(columns={"date": "日期", "open": "开盘", "close": "收盘",
                             "high": "最高", "low": "最低", "volume": "成交量"})
    df = df[["日期", "开盘", "收盘", "最高", "最低", "成交量"]].copy()
    df["日期"] = pd.to_datetime(df["日期"])
    df = df[df["日期"] >= pd.to_datetime(start)].reset_index(drop=True)
    _write_cache(path, df)
    return df


def fetch_stock_pool() -> pd.DataFrame:
    """沪深300 成分股列表（训练股票池）。"""
    path = _cache_path("pool", "hs300")
    cached = _read_cache(path)
    if cached is not None:
        return cached
    import akshare as ak
    raw = _retry(ak.index_stock_cons_csindex, symbol="000300")
    df = pd.DataFrame(raw)
    _write_cache(path, df)
    return df


# ---------------------------------------------------------------- 财报 ----

def fetch_financial(symbol: str) -> dict[str, pd.DataFrame]:
    """三大报表关键科目（新浪接口），返回 {资产负债表/利润表/现金流量表}。"""
    path = _cache_path("financial", symbol)
    cached = _read_cache(path)
    if cached is not None:
        return cached.to_dict("series") if False else _restore_financial(path)

    import akshare as ak
    try:
        bs = _retry(ak.stock_financial_report_sina, stock=symbol, symbol="资产负债表")
        income = _retry(ak.stock_financial_report_sina, stock=symbol, symbol="利润表")
        cash = _retry(ak.stock_financial_report_sina, stock=symbol, symbol="现金流量表")
    except Exception as exc:
        logger.warning("财报接口失败(%s)，尝试东财摘要接口", exc)
        try:
            summary = _retry(ak.stock_financial_abstract, symbol=symbol)
            result = {"财务摘要": summary}
            _write_cache(path, pd.DataFrame({"kind": ["财务摘要"]}))
            return result
        except Exception:
            logger.error("财报全部接口失败: %s", symbol)
            return {}
    result = {}
    for name, raw in [("资产负债表", bs), ("利润表", income), ("现金流量表", cash)]:
        if raw is not None and not raw.empty:
            df = raw.copy()
            if df.shape[0] < df.shape[1]:          # 新浪接口有时行列颠倒
                df = df.T
            if df.columns[0] not in ("报表日期", "类型"):
                df = df.T
            df = df.reset_index()
            df.columns = ["科目"] + [str(c) for c in df.columns[1:]]
            result[name] = df
    if result:
        _write_cache(path, pd.DataFrame({"kind": list(result.keys())}))
        for name, df in result.items():
            df.to_parquet(_cache_path(f"financial_{name}", symbol), index=False)
    return result


def _restore_financial(path: Path) -> dict[str, pd.DataFrame]:
    """从分文件缓存恢复财报数据。"""
    result: dict[str, pd.DataFrame] = {}
    for name in ("资产负债表", "利润表", "现金流量表"):
        p = _cache_path(f"financial_{name}", path.stem.split("_", 1)[1])
        if p.exists():
            result[name] = pd.read_parquet(p)
    return result


def fetch_financial_indicators(symbol: str) -> pd.DataFrame:
    """核心财务指标：ROE/毛利率/净利率/资产负债率等（东财主源，失败降级同花顺）。

    输出统一使用东财风格列名（净资产收益率(%) 等），与 data/finance.py 的 INDICATOR_MAP 对齐。
    """
    path = _cache_path("fin_ind", symbol)
    cached = _read_cache(path)
    if cached is not None:
        return cached

    df = _with_fallback(
        lambda: _fin_ind_em(symbol),
        lambda: _fin_ind_ths(symbol),
        backup_name="同花顺")
    _write_cache(path, df)
    return df


def _fin_ind_em(symbol: str) -> pd.DataFrame:
    """东财财务指标接口。"""
    import akshare as ak
    raw = _retry(ak.stock_financial_analysis_indicator, symbol=symbol, retries=1)
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    return df


# 同花顺字段 → 东财风格统一列名
_THS_INDICATOR_RENAME = {
    "报告期": "日期",
    "净资产收益率": "净资产收益率(%)",
    "销售毛利率": "销售毛利率(%)",
    "销售净利率": "销售净利率(%)",
    "资产负债率": "资产负债率(%)",
    "营业总收入同比增长率": "主营业务收入增长率(%)",
    "净利润同比增长率": "净利润增长率(%)",
    "基本每股收益": "每股收益_调整后(元)",
    "每股净资产": "每股净资产_调整后(元)",
}


def _fin_ind_ths(symbol: str) -> pd.DataFrame:
    """同花顺财务指标接口（备用，提供 ROE/毛利率/净利率/负债率等核心指标）。"""
    import akshare as ak
    raw = None
    for indicator in ("按报告期", "按年度"):   # 部分标的按报告期不可用则回退按年度
        try:
            raw = _retry(ak.stock_financial_abstract_ths, symbol=symbol,
                         indicator=indicator, retries=1)
            if raw is not None and not raw.empty:
                break
        except Exception as exc:
            logger.warning("同花顺(%s)失败: %s", indicator, exc)
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.rename(columns=_THS_INDICATOR_RENAME)
    cols = [c for c in _THS_INDICATOR_RENAME.values() if c in df.columns]
    out = df[cols].copy()
    out["日期"] = pd.to_datetime(out["日期"], errors="coerce")
    for c in cols:
        if c != "日期":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)


# ---------------------------------------------------------------- 舆情 ----

def fetch_news(symbol: str) -> pd.DataFrame:
    """个股新闻（东财接口），列：标题/内容/发布时间/来源/链接。"""
    path = _cache_path("news", symbol)
    cached = _read_cache(path)
    if cached is not None:
        return cached
    import akshare as ak
    raw = _retry(ak.stock_news_em, symbol=symbol)
    if raw is None or raw.empty:
        logger.warning("新闻为空: %s", symbol)
        return pd.DataFrame(columns=["标题", "内容", "发布时间", "来源", "链接"])
    rename = {"新闻标题": "标题", "新闻内容": "内容", "发布时间": "发布时间",
              "文章来源": "来源", "新闻链接": "链接"}
    df = raw.rename(columns=rename)
    cols = [c for c in ("标题", "内容", "发布时间", "来源", "链接") if c in df.columns]
    df = df[cols].copy()
    df["发布时间"] = pd.to_datetime(df["发布时间"], errors="coerce")
    df = df.drop_duplicates(subset=["标题"]).sort_values("发布时间", ascending=False)
    _write_cache(path, df)
    return df


# ---------------------------------------------------------------- 股票列表 ----

def fetch_stock_names() -> dict[str, str]:
    """全市场 A 股代码 → 名称映射（用于看板输入校验与名称显示）。

    接口失败返回空字典，调用方应退化为仅做格式校验。
    """
    path = _cache_path("stock_names", "all")
    cached = _read_cache(path)
    if cached is not None and "代码" in cached.columns:
        return dict(zip(cached["代码"].astype(str).str.zfill(6), cached["名称"].astype(str)))

    import akshare as ak
    try:
        raw = _retry(ak.stock_info_a_code_name, retries=1)
    except Exception as exc:
        logger.warning("股票列表接口失败(%s)，仅做格式校验", exc)
        return {}
    if raw is None or raw.empty:
        return {}
    df = pd.DataFrame(raw)
    if "code" in df.columns and "name" in df.columns:
        df = df.rename(columns={"code": "代码", "name": "名称"})
    if "代码" not in df.columns or "名称" not in df.columns:
        logger.warning("股票列表接口返回格式异常: %s", list(df.columns))
        return {}
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    _write_cache(path, df[["代码", "名称"]])
    return dict(zip(df["代码"], df["名称"].astype(str)))


# ---------------------------------------------------------------- 入口 ----

def summary(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "空数据"
    return f"{len(df)} 行 | 区间 {df['日期'].min()} ~ {df['日期'].max()}" if "日期" in df.columns else f"{len(df)} 行"


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "600519"
    for name, df in [("日K(前复权)", fetch_daily(sym)),
                     ("分钟线(5分钟)", fetch_minute(sym)),
                     ("新闻", fetch_news(sym))]:
        print(f"{name}: {summary(df)}")
    fin = fetch_financial(sym)
    print("财报报表:", list(fin.keys()) if fin else "无")
    ind = fetch_financial_indicators(sym)
    print(f"财务指标: {summary(ind)}")
