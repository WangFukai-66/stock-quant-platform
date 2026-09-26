"""P2A-02 / P2A-03 多模型融合引擎：Stacking 融合 + 市场状态动态权重。

防泄露设计：
- 基模型只在 train 上训练，在 val 上产生元特征（训练元学习器），test 上评估
- 动态权重：按市场状态（波动率分位 × 趋势方向）分象限，每象限在 val 上学习
  独立权重，test 上按当时状态选用
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from models.common import RESULTS_DIR, eval_classification, save_json

BASE_MODELS = ["xgb", "lstm", "transformer"]


# ---------------------------------------------------------------- 市场状态 ----

def market_state(close: pd.Series, vol_window: int = 20) -> pd.Series:
    """市场状态四象限：高波动/低波动 × 趋势/震荡 → 0~3。

    高波动: 20 日波动率处于近一年 70% 分位以上
    趋势:   收盘价 > MA60 且 MA20 斜率 > 0（或收盘 < MA60 且斜率 < 0，单向趋势）
    """
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


# ---------------------------------------------------------------- Stacking ----

class StackingFusion:
    """Stacking：基模型概率 → 元学习器（LogisticRegression）。"""

    def __init__(self, meta_learner=None):
        self.meta = meta_learner or LogisticRegression(max_iter=1000)
        self.weights: dict[str, float] = {}

    def fit(self, val_probs: dict[str, np.ndarray], val_y: np.ndarray):
        """val_probs: {模型名: 验证集概率数组}；必须与 val_y 同长。"""
        x_meta = np.column_stack([val_probs[m] for m in val_probs])
        self.meta.fit(x_meta, val_y)
        if hasattr(self.meta, "coef_"):
            total = float(np.abs(self.meta.coef_).sum()) + 1e-9
            self.weights = {m: float(w / total)
                            for m, w in zip(val_probs.keys(), np.abs(self.meta.coef_[0]))}
        return self

    def predict_proba(self, probs: dict[str, np.ndarray]) -> np.ndarray:
        x_meta = np.column_stack([probs[m] for m in probs])
        return self.meta.predict_proba(x_meta)[:, 1]


class DynamicWeightFusion:
    """动态权重：每个市场状态象限训练一个独立的 Stacking 元学习器。"""

    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.models: dict[int, StackingFusion] = {}
        self._names: list[str] | None = None

    def fit(self, val_probs: dict[str, np.ndarray], val_y: np.ndarray,
            val_states: np.ndarray):
        self._names = list(val_probs.keys())
        for s in range(self.n_states):
            mask = val_states == s
            if mask.sum() < 20:            # 样本不足的象限退化为等权
                continue
            sub_probs = {m: p[mask] for m, p in val_probs.items()}
            self.models[s] = StackingFusion().fit(sub_probs, val_y[mask])
        return self

    def predict_proba(self, probs: dict[str, np.ndarray], states: np.ndarray) -> np.ndarray:
        out = np.full(len(states), 0.5)
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() == 0:
                continue
            if s in self.models:
                sub = {m: p[mask] for m, p in probs.items()}
                out[mask] = self.models[s].predict_proba(sub)
            else:
                # 等权平均兜底
                out[mask] = np.mean([probs[m][mask] for m in probs], axis=0)
        return out


# ---------------------------------------------------------------- 对比评估 ----

def compare_models(y_true: np.ndarray, probs: dict[str, np.ndarray]) -> pd.DataFrame:
    """单模型 / 等权平均 / Stacking / 动态权重 对比表。"""
    rows = []
    for name, p in probs.items():
        m = eval_classification(y_true, p)
        rows.append({"模型": name, "accuracy": m["accuracy"], "auc": m["auc"]})
    avg = np.mean([p for p in probs.values()], axis=0)
    m = eval_classification(y_true, avg)
    rows.append({"模型": "等权平均", "accuracy": m["accuracy"], "auc": m["auc"]})
    return pd.DataFrame(rows)


def run_fusion_pipeline(val_probs: dict[str, np.ndarray], val_y: np.ndarray,
                        test_probs: dict[str, np.ndarray], test_y: np.ndarray,
                        val_states: np.ndarray | None = None,
                        test_states: np.ndarray | None = None,
                        suffix: str = "") -> dict:
    """完整融合流程：Stacking + 动态权重 + 对比表 + 落盘。

    suffix: 输出文件名后缀（如 "_600519"），按股票区分避免多标的互相覆盖。
    """
    fusion = StackingFusion().fit(val_probs, val_y)
    fused_prob = fusion.predict_proba(test_probs)
    fusion_metrics = eval_classification(test_y, fused_prob)
    print(f"[Stacking 融合] accuracy={fusion_metrics['accuracy']} auc={fusion_metrics['auc']}")
    print(f"元学习器权重: { {k: round(v, 3) for k, v in fusion.weights.items()} }")

    dyn_metrics = None
    if val_states is not None and test_states is not None:
        dyn = DynamicWeightFusion().fit(val_probs, val_y, val_states)
        dyn_prob = dyn.predict_proba(test_probs, test_states)
        dyn_metrics = eval_classification(test_y, dyn_prob)
        print(f"[动态权重融合] accuracy={dyn_metrics['accuracy']} auc={dyn_metrics['auc']}")

    compare = compare_models(test_y, test_probs)
    stacking_row = pd.DataFrame([{"模型": "Stacking融合", "accuracy": fusion_metrics["accuracy"],
                                  "auc": fusion_metrics["auc"]}])
    compare = pd.concat([compare, stacking_row], ignore_index=True)
    if dyn_metrics:
        compare = pd.concat([compare, pd.DataFrame(
            [{"模型": "动态权重融合", "accuracy": dyn_metrics["accuracy"],
              "auc": dyn_metrics["auc"]}])], ignore_index=True)
    compare.to_csv(RESULTS_DIR / f"fusion_compare{suffix}.csv", index=False)
    print(compare.to_string(index=False))

    result = {"stacking": fusion_metrics, "dynamic": dyn_metrics,
              "weights": fusion.weights, "compare": compare.to_dict("records")}
    # 顶层 accuracy/auc 便于看板直接展示（find_metrics 读取顶层字段）
    result["accuracy"] = fusion_metrics["accuracy"]
    result["auc"] = fusion_metrics["auc"]
    save_json(result, RESULTS_DIR / f"fusion_metrics{suffix}.json")
    return result
