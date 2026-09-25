"""P1-02 财报模块：核心财务指标提取与展示。"""
from __future__ import annotations

import pandas as pd

from data.fetcher import fetch_financial, fetch_financial_indicators

INDICATOR_MAP = {
    "净资产收益率(%)": "ROE",
    "销售毛利率(%)": "毛利率",
    "销售净利率(%)": "净利率",
    "资产负债率(%)": "资产负债率",
    "主营业务收入增长率(%)": "营收增长率",
    "净利润增长率(%)": "净利润增长率",
    "每股收益_调整后(元)": "每股收益",
    "每股净资产_调整后(元)": "每股净资产",
}


def get_financial_metrics(symbol: str, years: int = 5) -> pd.DataFrame:
    """核心财务指标时间序列（东财指标接口，按报告期排序）。"""
    raw = fetch_financial_indicators(symbol)
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    cols = [c for c in INDICATOR_MAP if c in df.columns]
    if not cols:
        return pd.DataFrame()
    out = df[["日期"] + cols].dropna(subset=["日期"]).sort_values("日期")
    out = out.rename(columns=INDICATOR_MAP)
    return out.tail(years * 4).reset_index(drop=True)   # 季报，5年≈20期


def get_report_summary(symbol: str, report: str = "资产负债表", top: int = 8) -> pd.DataFrame:
    """三大报表关键科目摘要（新浪接口）。"""
    reports = fetch_financial(symbol)
    if report not in reports:
        return pd.DataFrame()
    df = reports[report]
    return df.head(top)
