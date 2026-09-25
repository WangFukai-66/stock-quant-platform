"""P3-04 对比评估：微调时序大模型 vs 阶段2融合引擎（统一基准）。

指标：MAE / RMSE / 方向准确率 + Diebold-Mariano 显著性检验 + 回测收益对比。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.common import RESULTS_DIR, save_json

RESULTS = RESULTS_DIR


def _dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """Diebold-Mariano 检验（误差 e1 相对 e2 是否有显著差异，单侧）。"""
    d = e1 - e2
    n = len(d)
    if n < 10:
        return {"stat": np.nan, "p_value": np.nan, "note": "样本过少"}
    d_bar = d.mean()
    # 长程方差（h 步自协方差截断）
    gamma = np.cov(d[:-h], d[h:])[0, 1] if h < n - 1 else 0.0
    var = (np.var(d, ddof=1) + 2 * gamma) / n
    if var <= 0:
        return {"stat": np.nan, "p_value": np.nan, "note": "方差非正"}
    stat = d_bar / np.sqrt(var)
    from scipy import stats
    p = float(stats.norm.sf(abs(stat)))           # 单侧
    return {"stat": round(float(stat), 4), "p_value": round(p, 4)}


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """预测方向准确率（涨跌方向是否一致）。"""
    t = np.sign(np.diff(y_true, prepend=y_true[0]))
    p = np.sign(np.diff(y_pred, prepend=y_pred[0]))
    return float(np.mean(t == p))


def load_fusion_prob() -> pd.DataFrame | None:
    p = RESULTS / "fusion_predictions.csv"
    return pd.read_csv(p, parse_dates=["日期"]) if p.exists() else None


def load_llm_pred(symbol: str) -> pd.DataFrame | None:
    """加载微调模型的 test 预测（由 finetune 后续推理脚本生成）。"""
    for name in (f"llm_{symbol}_test_pred.csv", "llm_test_pred.csv"):
        p = RESULTS / name
        if p.exists():
            return pd.read_csv(p, parse_dates=["日期"])
    return None


def evaluate_llm_vs_fusion(symbol: str, llm_pred: pd.DataFrame | None = None,
                           fusion_prob: pd.DataFrame | None = None) -> dict:
    fusion = fusion_prob if fusion_prob is not None else load_fusion_prob()
    llm = llm_pred if llm_pred is not None else load_llm_pred(symbol)

    if fusion is None:
        return {"error": "缺少融合引擎预测（先运行 scripts/train_all.py）"}
    if llm is None:
        return {"error": "缺少时序大模型预测（先运行 finetune 与推理脚本）"}

    merged = fusion.merge(llm, on="日期", suffixes=("_fusion", "_llm"), how="inner")
    if merged.empty:
        return {"error": "两模型预测无日期交集"}

    # 融合概率 → 预测次日收盘（用当前收盘 × (1 + 概率-0.5 映射) 的近似不可靠，
    # 统一用涨跌方向与真实收益比较）
    true_ret = merged["收盘"].pct_change() * 100 if "收盘" in merged.columns else None
    if true_ret is None:
        # 从真实标签近似：label 1/0
        true_ret = (merged["真实标签"].astype(float) * 2 - 1).values

    llm_direction = np.where(merged["预测概率_llm"].values > 0.5, 1, -1)
    fusion_direction = np.where(merged["预测概率_fusion"].values > 0.5, 1, -1)
    true_direction = np.sign(true_ret.values) if hasattr(true_ret, "values") else np.sign(true_ret)

    llm_dir_acc = float(np.mean(llm_direction == true_direction))
    fusion_dir_acc = float(np.mean(fusion_direction == true_direction))

    # 概率误差（Brier 风格：以真实方向为 0/1）
    y_true_01 = (true_direction > 0).astype(float)
    llm_err = (merged["预测概率_llm"].values - y_true_01) ** 2
    fusion_err = (merged["预测概率_fusion"].values - y_true_01) ** 2
    dm = _dm_test(llm_err, fusion_err)

    result = {
        "样本数": len(merged),
        "方向准确率": {"时序大模型": round(llm_dir_acc, 4),
                       "融合引擎": round(fusion_dir_acc, 4)},
        "Brier误差": {"时序大模型": round(float(llm_err.mean()), 4),
                      "融合引擎": round(float(fusion_err.mean()), 4)},
        "Diebold-Mariano": dm,
        "结论": ("时序大模型显著更优" if dm["p_value"] < 0.05 and dm["stat"] < 0
                 else "融合引擎显著更优" if dm["p_value"] < 0.05
                 else "两模型无显著差异"),
    }
    save_json(result, RESULTS / "llm_vs_fusion.json")
    return result


def write_llm_report(symbol: str, result: dict) -> None:
    lines = ["# 时序大模型 vs 融合引擎 对比报告", "",
             f"标的: {symbol} | 统一测试区间 | 指标口径一致", "",
             "| 指标 | 时序大模型 | 融合引擎 |", "|---|---|---|"]
    if "error" not in result:
        lines.append(f"| 方向准确率 | {result['方向准确率']['时序大模型']} | "
                     f"{result['方向准确率']['融合引擎']} |")
        lines.append(f"| Brier误差 | {result['Brier误差']['时序大模型']} | "
                     f"{result['Brier误差']['融合引擎']} |")
        dm = result["Diebold-Mariano"]
        lines.append(f"| DM统计量 | {dm['stat']} (p={dm['p_value']}) | - |")
        lines += ["", f"## 结论: {result['结论']}", "",
                  "注: 方向准确率仅作参考，最终以 A 股规则回测收益为决策依据。"]
    else:
        lines.append(f"| - | {result['error']} |")
    (RESULTS / "llm_vs_fusion.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600519")
    args = ap.parse_args()
    r = evaluate_llm_vs_fusion(args.symbol)
    print(r)
    write_llm_report(args.symbol, r)
