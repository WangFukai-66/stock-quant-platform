"""训练全部模型（XGBoost + LSTM + Transformer）并执行 Stacking / 动态权重融合。

用法: python scripts/train_all.py --symbol 600519 [--epochs 30]
输出: results/ 下各模型指标与融合对比表、test 集预测
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from models.common import RESULTS_DIR, save_predictions
from models.fusion import market_state, run_fusion_pipeline
from scripts.pipeline import prepare_sequence, prepare_tabular


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600519")
    ap.add_argument("--start", default="20150101")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--window", type=int, default=60)
    args = ap.parse_args()

    # ---- 表格数据（XGBoost）----
    train, val, test, _ = prepare_tabular(args.symbol, start=args.start)

    # ---- 序列数据（LSTM / Transformer）----
    (x_tr, y_tr, _, _), (x_va, y_va, d_va, _), (x_te, y_te, d_te, c_te), _ = \
        prepare_sequence(args.symbol, window=args.window, start=args.start)

    # 序列窗口比表格数据短 window 行，用日期对齐（窗口结束日与表格行日期一致）
    def _align_mask(dates_df, dates_seq):
        return np.isin(dates_df.astype("datetime64[ns]"), dates_seq.astype("datetime64[ns]"))

    val_mask = _align_mask(val["日期"].values, d_va)
    test_mask = _align_mask(test["日期"].values, d_te)

    val_probs: dict[str, np.ndarray] = {}
    test_probs: dict[str, np.ndarray] = {}

    # ---- XGBoost ----
    print("\n[1/3] XGBoost")
    from models.xgboost_model import XGBBaseline
    xgb = XGBBaseline()
    xgb.fit(train, val, verbose=False)
    val_probs["xgb"] = xgb.predict_proba(val).values[val_mask]
    test_probs["xgb"] = xgb.predict_proba(test).values[test_mask]
    xgb.save()
    save_predictions(test["日期"].values, test["label"].values, xgb.predict_proba(test).values,
                     test["收盘"].values, RESULTS_DIR / "xgb_predictions.csv")

    # ---- LSTM ----
    print("\n[2/3] LSTM")
    from models.lstm_model import predict_proba as lstm_prob, train_lstm
    lstm = train_lstm(x_tr, y_tr, x_va, y_va, epochs=args.epochs)
    val_probs["lstm"] = lstm_prob(lstm, x_va)
    test_probs["lstm"] = lstm_prob(lstm, x_te)
    save_predictions(d_te, y_te, test_probs["lstm"], c_te,
                     RESULTS_DIR / "lstm_predictions.csv")

    # ---- Transformer ----
    print("\n[3/3] Transformer")
    from models.transformer_model import (predict_proba as tf_prob, train_transformer)
    tf = train_transformer(x_tr, y_tr, x_va, y_va, epochs=args.epochs)
    val_probs["transformer"] = tf_prob(tf, x_va)
    test_probs["transformer"] = tf_prob(tf, x_te)
    save_predictions(d_te, y_te, test_probs["transformer"], c_te,
                     RESULTS_DIR / "transformer_predictions.csv")

    # ---- 融合 ----
    print("\n[融合] Stacking + 动态权重")
    val_states = market_state(pd.Series(val["收盘"].values)).values[val_mask]
    test_states = market_state(pd.Series(test["收盘"].values)).values[test_mask]
    result = run_fusion_pipeline(val_probs, val["label"].values[val_mask],
                                 test_probs, test["label"].values[test_mask],
                                 val_states=val_states, test_states=test_states)

    # 融合预测落盘（供看板/回测优先使用；日期与序列窗口对齐）
    from models.fusion import StackingFusion
    fusion = StackingFusion().fit(val_probs, val["label"].values[val_mask])
    fused = fusion.predict_proba(test_probs)
    save_predictions(d_te, y_te, fused, c_te, RESULTS_DIR / "fusion_predictions.csv")
    print("\n全部完成，启动看板: streamlit run app/main.py")


if __name__ == "__main__":
    main()
