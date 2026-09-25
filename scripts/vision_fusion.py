"""P2C-03 视觉信号融合：CNN 形态概率作为第 4 路信号加入 Stacking。

对比：三模型融合（XGB+LSTM+Transformer） vs 四路信号融合（+CNN 形态概率）。
注意：CNN 使用合成形态训练，对真实 K 线属于零样本迁移演示；结论需谨慎解读。

用法: python scripts/vision_fusion.py --symbol 600519 [--epochs 20] [--cnn-epochs 5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data.quote import get_history
from factors.process import build_dataset, split_temporal
from models.common import RESULTS_DIR, eval_classification
from models.fusion import StackingFusion
from scripts.pipeline import prepare_sequence, prepare_tabular


def render_windows_for_dates(ohlc: pd.DataFrame, dates: pd.Series, window: int = 60):
    """为指定日期渲染其前 60 日 K 线窗口（对齐 CNN 训练图的窗口长度）。"""
    from vision.render import render_candlestick
    o = ohlc.reset_index(drop=True)
    pos = {d: i for i, d in enumerate(o["日期"])}
    images = []
    for d in dates:
        i = pos.get(d)
        if i is None or i < window - 1:
            images.append(None)
        else:
            images.append(render_candlestick(o.iloc[i - window + 1:i + 1]))
    return images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600519")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--cnn-epochs", type=int, default=5)
    args = ap.parse_args()

    print("=" * 60, "\n[1/4] 训练三基模型")
    train, val, test, _ = prepare_tabular(args.symbol)
    (x_tr, y_tr, _, _), (x_va, y_va, d_va, _), (x_te, y_te, d_te, c_te), _ = \
        prepare_sequence(args.symbol)

    def _align_mask(dates_df, dates_seq):
        return np.isin(dates_df.astype("datetime64[ns]"), dates_seq.astype("datetime64[ns]"))

    val_mask = _align_mask(val["日期"].values, d_va)
    test_mask = _align_mask(test["日期"].values, d_te)

    from models.xgboost_model import XGBBaseline
    xgb = XGBBaseline().fit(train, val, verbose=False)
    val_probs = {"xgb": xgb.predict_proba(val).values[val_mask]}
    test_probs = {"xgb": xgb.predict_proba(test).values[test_mask]}

    from models.lstm_model import predict_proba as lstm_prob, train_lstm
    lstm = train_lstm(x_tr, y_tr, x_va, y_va, epochs=args.epochs)
    val_probs["lstm"] = lstm_prob(lstm, x_va)
    test_probs["lstm"] = lstm_prob(lstm, x_te)

    from models.transformer_model import predict_proba as tf_prob, train_transformer
    tf = train_transformer(x_tr, y_tr, x_va, y_va, epochs=args.epochs)
    val_probs["transformer"] = tf_prob(tf, x_va)
    test_probs["transformer"] = tf_prob(tf, x_te)

    val_y = val["label"].values[val_mask]
    test_y = test["label"].values[test_mask]

    print("=" * 60, "\n[2/4] 三模型 Stacking 融合（基线）")
    base_fusion = StackingFusion().fit(val_probs, val_y)
    base_prob = base_fusion.predict_proba(test_probs)
    base_metrics = eval_classification(test_y, base_prob)
    print(f"三模型融合: acc={base_metrics['accuracy']} auc={base_metrics['auc']}")

    print("=" * 60, "\n[3/4] CNN 形态识别（合成数据训练）")
    from vision.cnn_pattern import predict_pattern_probs, train_pattern_cnn
    from vision.render import build_synthetic_dataset
    imgs, labs = build_synthetic_dataset(n_per_class=120)
    cnn = train_pattern_cnn(imgs, labs, epochs=args.cnn_epochs)

    print("渲染 val/test 窗口 K 线图...")
    ohlc = get_history(args.symbol, period="daily")
    val_imgs = render_windows_for_dates(ohlc, val["日期"])
    test_imgs = render_windows_for_dates(ohlc, test["日期"])
    val_valid = [i for i, im in enumerate(val_imgs) if im is not None]
    test_valid = [i for i, im in enumerate(test_imgs) if im is not None]

    # CNN 概率对齐到序列窗口日期（与三模型概率同长）
    val_cnn_full = np.full((len(val), 5), 0.2)
    test_cnn_full = np.full((len(test), 5), 0.2)
    if val_valid:
        val_cnn_full[val_valid] = predict_pattern_probs(cnn, [val_imgs[i] for i in val_valid])
    if test_valid:
        test_cnn_full[test_valid] = predict_pattern_probs(cnn, [test_imgs[i] for i in test_valid])
    val_cnn = val_cnn_full[val_mask]
    test_cnn = test_cnn_full[test_mask]

    print("=" * 60, "\n[4/4] 四路信号融合（+CNN 形态概率）")
    val_meta = np.column_stack([val_probs["xgb"], val_probs["lstm"],
                                val_probs["transformer"], val_cnn])
    test_meta = np.column_stack([test_probs["xgb"], test_probs["lstm"],
                                 test_probs["transformer"], test_cnn])
    from sklearn.linear_model import LogisticRegression
    meta = LogisticRegression(max_iter=1000)
    meta.fit(val_meta, val_y)
    vision_prob = meta.predict_proba(test_meta)[:, 1]
    vision_metrics = eval_classification(test_y, vision_prob)
    print(f"四路信号融合: acc={vision_metrics['accuracy']} auc={vision_metrics['auc']}")

    lines = ["# 视觉信号融合报告", "",
             "## 对比", "",
             f"| 方案 | accuracy | AUC |", "|---|---|---|",
             f"| 三模型 Stacking 融合 | {base_metrics['accuracy']} | {base_metrics['auc']} |",
             f"| 四路信号融合(+CNN形态) | {vision_metrics['accuracy']} | {vision_metrics['auc']} |",
             "", "## 结论", ""]
    delta = vision_metrics["auc"] - base_metrics["auc"]
    if delta > 0.005:
        lines.append(f"CNN 形态信号带来 +{delta:.4f} AUC 增量，视觉信号具备融合价值。")
    else:
        lines.append(f"CNN 形态信号增量 {delta:+.4f}，未显著超越三模型融合。"
                     "可能原因：CNN 用合成形态训练、与真实 K 线存在分布差异，"
                     "建议补充人工标注真实 K 线形态后重试。")
    lines += ["", "## 说明", "",
              "- CNN 训练数据为合成形态（头肩顶/双底/上升三角/箱体/突破）",
              "- 真实 K 线窗口为 60 日，与合成样本长度一致",
              "- 本实验为零样本迁移演示，论文中需注明训练域与测试域差异"]
    report = "\n".join(lines)
    (RESULTS_DIR / "vision_fusion_report.md").write_text(report, encoding="utf-8")
    print("\n报告已保存: results/vision_fusion_report.md")


if __name__ == "__main__":
    main()
