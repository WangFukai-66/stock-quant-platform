"""全局视觉样式：白色清爽风 CSS + 品牌横幅（纯 CSS/HTML，零依赖，云端安全）。"""
from __future__ import annotations

import streamlit as st

CSS = """
<style>
/* ---------- 全局背景：白色 + 极淡红/蓝晕点缀 ---------- */
.stApp {
    background:
        radial-gradient(1100px 500px at 85% -10%, rgba(230, 69, 69, 0.045), transparent 60%),
        radial-gradient(900px 420px at -10% 110%, rgba(37, 99, 235, 0.04), transparent 55%),
        #ffffff;
}

/* ---------- 标题：h1 左侧红色竖线装饰 ---------- */
h1 {
    padding-left: 0.75rem;
    border-left: 4px solid #e64545;
    letter-spacing: 0.02em;
}
h2 { letter-spacing: 0.015em; }
h3 { letter-spacing: 0.01em; }

/* ---------- 指标卡：白卡 + 浅边框 + 悬停微动效 ---------- */
[data-testid="stMetric"] {
    background: linear-gradient(160deg, #ffffff 0%, #fafbfc 100%);
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    transition: border-color .25s ease, box-shadow .25s ease, transform .25s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(230, 69, 69, 0.45);
    box-shadow: 0 6px 18px rgba(230, 69, 69, 0.10);
    transform: translateY(-2px);
}
[data-testid="stMetricLabel"] { color: #6b7280; }

/* ---------- 侧边栏：浅灰底 + 细分隔线 ---------- */
[data-testid="stSidebar"] {
    background: #fafbfc;
    border-right: 1px solid #eceef1;
}

/* ---------- 页签 radio：hover 高亮 + 选中态红条 ---------- */
[data-testid="stSidebar"] [role="radiogroup"] label {
    border-radius: 8px;
    padding: 0.35rem 0.5rem;
    transition: background .2s ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(230, 69, 69, 0.06);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: rgba(230, 69, 69, 0.10);
    box-shadow: inset 3px 0 0 #e64545;
}

/* ---------- 数据表格：统一边框 ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    overflow: hidden;
}

/* ---------- 输入控件圆角 ---------- */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div,
[data-testid="stDateInput"] > div {
    border-radius: 8px;
}

/* ---------- 自定义滚动条 ---------- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #f3f4f6; }
::-webkit-scrollbar-thumb {
    background: #d1d5db;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover { background: #e64545; }

/* ---------- 页面切换：重跑期间保持旧内容可见，避免整页淡白闪烁 ---------- */
[data-stale="true"],
.stale-element {
    opacity: 1 !important;
}

/* ---------- 品牌横幅：白底 + 红色渐变点缀 ---------- */
.cy-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.8rem;
    padding: 1.15rem 1.4rem;
    margin: 0 0 1.2rem 0;
    border-radius: 14px;
    border: 1px solid rgba(230, 69, 69, 0.28);
    background:
        linear-gradient(115deg, #ffffff 0%, #fdf6f6 55%, #fdecec 100%);
    box-shadow: 0 6px 20px rgba(230, 69, 69, 0.08);
}
.cy-hero-title {
    font-size: 1.32rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    color: #1a1d24;
    margin: 0 0 0.25rem 0;
}
.cy-hero-sub {
    font-size: 0.82rem;
    color: #6b7280;
    margin: 0;
}
.cy-hero-pills { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.cy-pill {
    font-size: 0.75rem;
    padding: 0.28rem 0.75rem;
    border-radius: 999px;
    border: 1px solid;
    white-space: nowrap;
}
.cy-pill-red   { color: #c53030; border-color: rgba(230, 69, 69, 0.5);  background: rgba(230, 69, 69, 0.08); }
.cy-pill-blue  { color: #1d4ed8; border-color: rgba(59, 130, 246, 0.5); background: rgba(59, 130, 246, 0.08); }
.cy-pill-gold  { color: #b45309; border-color: rgba(245, 158, 11, 0.5); background: rgba(245, 158, 11, 0.10); }
</style>
"""


# Plotly 图表统一配置：滚轮 / 触控板 / 触屏双指缩放 + 隐藏 plotly 图标
PLOTLY_CONFIG = {"scrollZoom": True, "displaylogo": False}


def inject_style() -> None:
    """注入全局自定义 CSS（在 set_page_config 之后调用一次）。"""
    st.markdown(CSS, unsafe_allow_html=True)


def hero() -> None:
    """紧凑品牌横幅：项目标题 + pill 标签，展示于所有页签顶部。"""
    st.markdown(
        """
        <div class="cy-hero">
            <div>
                <div class="cy-hero-title">基于多模型融合的 A 股量化回测与智能交易预警系统</div>
                <p class="cy-hero-sub">雏雁计划 · 国际学院 智能科学与技术专业 · 数据驱动 · 理性决策</p>
            </div>
            <div class="cy-hero-pills">
                <span class="cy-pill cy-pill-red">LSTM · XGBoost · Transformer</span>
                <span class="cy-pill cy-pill-blue">Stacking · 动态权重</span>
                <span class="cy-pill cy-pill-gold">T+1 · 涨跌停规则回测</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
