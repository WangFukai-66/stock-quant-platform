"""P1-03 / P2B-01 舆情情绪分析。

双模式：
- 规则模式（默认可用）：金融情感词典打分，输出 [-1,1] 与三分类
- 深度模式：chinese-roberta-wwm-ext 三分类（首次运行自动下载约 400MB，CPU 可推理）
  加载失败自动降级为规则模式。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

POSITIVE_WORDS = {
    "增长", "上涨", "盈利", "利好", "突破", "创新高", "中标", "回购", "增持", "分红",
    "预增", "扭亏", "超预期", "涨停", "受益", "获批", "签约", "扩产", "提价", "降本",
    "改善", "提升", "看好", "买入", "推荐", "加仓", "回暖", "强劲", "大涨", "涨停板",
}
NEGATIVE_WORDS = {
    "下跌", "亏损", "利空", "减持", "违规", "处罚", "退市", "风险", "暴跌", "预亏",
    "下滑", "低于预期", "跌停", "商誉减值", "质押", "诉讼", "停产", "召回", "事故",
    "裁员", "债务", "违约", "爆雷", "造假", "监管", "警示", "冻结", "破产", "利空出尽",
}


def rule_sentiment(text: str) -> tuple[float, str]:
    """规则情感打分：返回 (分数∈[-1,1], 分类)。"""
    if not text or not isinstance(text, str):
        return 0.0, "中性"
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    if pos + neg == 0:
        return 0.0, "中性"
    score = (pos - neg) / (pos + neg)
    if score > 0.15:
        return round(score, 3), "正面"
    if score < -0.15:
        return round(score, 3), "负面"
    return 0.0, "中性"


def _clean(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()[:256]


class RobertaSentiment:
    """深度模式：中文 RoBERTa 情绪三分类。"""

    LABELS = ["负面", "中性", "正面"]

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            name = "hfl/chinese-roberta-wwm-ext"
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                name, num_labels=3, ignore_mismatched_sizes=True)
            self.model.eval()
            logger.info("RoBERTa 情绪模型加载成功")
        except Exception as exc:
            logger.warning("深度情绪模型加载失败(%s)，降级为规则打分", exc)
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, texts: list[str]) -> list[tuple[float, str]]:
        """返回 [(分数∈[-1,1], 分类), ...]。"""
        if not self.available:
            return [rule_sentiment(t) for t in texts]
        import torch
        cleaned = [_clean(t) for t in texts]
        enc = self.tokenizer(cleaned, padding=True, truncation=True, max_length=128,
                             return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)          # 0负 1中 2正
        results = []
        for p in probs:
            score = float(p[2] - p[0])
            label = self.LABELS[int(p.argmax())]
            results.append((round(score, 3), label))
        return results


_sentiment_model: RobertaSentiment | None = None


def get_sentiment_model(use_deep: bool = True) -> RobertaSentiment:
    global _sentiment_model
    if _sentiment_model is None:
        _sentiment_model = RobertaSentiment()
    if not use_deep and _sentiment_model.available is False:
        return _sentiment_model
    return _sentiment_model


def analyze_news(df: pd.DataFrame, use_deep: bool = False) -> pd.DataFrame:
    """新闻 DataFrame → 情绪分/分类 + 每日热度。"""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    texts = out["标题"].fillna("") + " " + out.get("内容", pd.Series("", index=out.index)).fillna("")
    if use_deep:
        model = get_sentiment_model()
        scored = model.predict(texts.tolist())
        out["情绪分"] = [s for s, _ in scored]
        out["情绪"] = [c for _, c in scored]
    else:
        scored = texts.map(rule_sentiment)
        out["情绪分"] = [s for s, _ in scored]
        out["情绪"] = [c for _, c in scored]

    if "发布时间" in out.columns:
        daily = out.dropna(subset=["发布时间"]).groupby(out["发布时间"].dt.date).agg(
            新闻数=("标题", "count"), 情绪均值=("情绪分", "mean")).reset_index()
        return out, daily.rename(columns={"发布时间": "日期"})
    return out, pd.DataFrame()
