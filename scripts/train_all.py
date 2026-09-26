"""训练全部模型（XGBoost + LSTM + Transformer）并执行 Stacking / 动态权重融合。

用法:
    python -m scripts.train_all --symbol 600519              # 标准模式（30 epochs）
    python -m scripts.train_all --symbol 600519 --quick      # 快速模式（10 epochs，约 3~6 分钟）
    python -m scripts.train_all --symbol 600519 --status results/training_600519.json

输出:
    - results/ 下各模型指标与融合对比表、test 集预测（文件名带股票代码后缀，
      如 fusion_predictions_600519.csv，供看板按标的读取，避免不同股票预测张冠李戴）
    - models/ 下模型权重按股票保存（xgb_model_600519.json、lstm_600519.pt 等）
    - stdout 输出 "PROGRESS <0-100>" 标记行，供看板后台训练展示进度
    - --status 指定时写入训练状态文件（status: running/done/error + progress + pid）
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from models.common import (MODELS_DIR, RESULTS_DIR, eval_classification,
                           save_json, save_predictions)
from models.fusion import market_state, run_fusion_pipeline
from scripts.pipeline import prepare_sequence, prepare_tabular

# 最少交易日样本数（build_dataset 已截断 60 日指标预热期），不足则拒绝训练
MIN_SAMPLES = 300


def write_status(path: Path | None, info: dict) -> None:
    """训练状态文件写入（path 为 None 时跳过，命令行直接使用无副作用）。"""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600519")
    ap.add_argument("--start", default="20150101")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--quick", action="store_true", help="快速模式：epochs 降为 10，约 3~6 分钟")
    ap.add_argument("--status", default=None, help="训练状态文件路径（供看板后台训练读取）")
    args = ap.parse_args()

    epochs = 10 if args.quick else args.epochs
    status_path = Path(args.status) if args.status else None

    def set_status(status: str, progress: int, error: str | None = None) -> None:
        write_status(status_path, {
            "symbol": args.symbol,
            "mode": "quick" if args.quick else "standard",
            "status": status,
            "progress": progress,
            "error": error,
            "pid": os.getpid(),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })

    def progress(pct: int) -> None:
        print(f"PROGRESS {pct}", flush=True)
        set_status("running", pct)

    set_status("running", 0)
    try:
        # ---- 数据准备（0~5）----
        print(f"[数据] 准备 {args.symbol} 训练数据（模式={'快速' if args.quick else '标准'}，"
              f"epochs={epochs}）")
        train, val, test, _ = prepare_tabular(args.symbol, start=args.start)
        progress(2)
        (x_tr, y_tr, _, _), (x_va, y_va, d_va, _), (x_te, y_te, d_te, c_te), _ = \
            prepare_sequence(args.symbol, window=args.window, start=args.start)
        progress(4)

        # ---- 数据量校验（防次新股样本不足导致崩溃）----
        total = len(train) + len(val) + len(test)
        if total < MIN_SAMPLES:
            raise RuntimeError(
                f"历史数据不足：仅 {total} 个交易日（含预热期需至少约 {MIN_SAMPLES * 2} 个），"
                f"该股票上市时间过短，无法训练")
        if len(val) <= args.window or len(test) <= args.window:
            raise RuntimeError(
                f"验证/测试集样本不足（{len(val)}/{len(test)} 行，需大于滑窗 {args.window} 行）")

        # 序列窗口比表格数据短 window 行，用日期对齐（窗口结束日与表格行日期一致）
        def _align_mask(dates_df, dates_seq):
            return np.isin(dates_df.astype("datetime64[ns]"), dates_seq.astype("datetime64[ns]"))

        val_mask = _align_mask(val["日期"].values, d_va)
        test_mask = _align_mask(test["日期"].values, d_te)

        val_probs: dict[str, np.ndarray] = {}
        test_probs: dict[str, np.ndarray] = {}

        # ---- XGBoost（5~25）----
        print("\n[1/3] XGBoost")
        from models.xgboost_model import XGBBaseline
        xgb = XGBBaseline(model_path=MODELS_DIR / f"xgb_model_{args.symbol}.json")
        xgb.fit(train, val, verbose=False)
        progress(25)
        val_probs["xgb"] = xgb.predict_proba(val).values[val_mask]
        test_probs["xgb"] = xgb.predict_proba(test).values[test_mask]
        xgb.save()
        save_predictions(test["日期"].values, test["label"].values, xgb.predict_proba(test).values,
                         test["收盘"].values, RESULTS_DIR / f"xgb_predictions_{args.symbol}.csv")
        save_json(eval_classification(test["label"].values[test_mask], test_probs["xgb"]),
                  RESULTS_DIR / f"xgb_metrics_{args.symbol}.json")

        # ---- LSTM（25~50）----
        print("\n[2/3] LSTM")
        from models.lstm_model import predict_proba as lstm_prob, train_lstm

        def _lstm_cb(epoch: int, total_epochs: int) -> None:
            progress(25 + int(25 * epoch / total_epochs))

        lstm = train_lstm(x_tr, y_tr, x_va, y_va, epochs=epochs,
                          model_path=MODELS_DIR / f"lstm_{args.symbol}.pt",
                          progress_cb=_lstm_cb)
        progress(50)
        val_probs["lstm"] = lstm_prob(lstm, x_va)
        test_probs["lstm"] = lstm_prob(lstm, x_te)
        save_predictions(d_te, y_te, test_probs["lstm"], c_te,
                         RESULTS_DIR / f"lstm_predictions_{args.symbol}.csv")
        save_json(eval_classification(y_te, test_probs["lstm"]),
                  RESULTS_DIR / f"lstm_metrics_{args.symbol}.json")

        # ---- Transformer（50~75）----
        print("\n[3/3] Transformer")
        from models.transformer_model import (predict_proba as tf_prob, train_transformer)

        def _tf_cb(epoch: int, total_epochs: int) -> None:
            progress(50 + int(25 * epoch / total_epochs))

        tf = train_transformer(x_tr, y_tr, x_va, y_va, epochs=epochs,
                               model_path=MODELS_DIR / f"transformer_{args.symbol}.pt",
                               progress_cb=_tf_cb)
        progress(75)
        val_probs["transformer"] = tf_prob(tf, x_va)
        test_probs["transformer"] = tf_prob(tf, x_te)
        save_predictions(d_te, y_te, test_probs["transformer"], c_te,
                         RESULTS_DIR / f"transformer_predictions_{args.symbol}.csv")
        save_json(eval_classification(y_te, test_probs["transformer"]),
                  RESULTS_DIR / f"transformer_metrics_{args.symbol}.json")

        # ---- 融合（75~100）----
        print("\n[融合] Stacking + 动态权重")
        val_states = market_state(pd.Series(val["收盘"].values)).values[val_mask]
        test_states = market_state(pd.Series(test["收盘"].values)).values[test_mask]
        run_fusion_pipeline(val_probs, val["label"].values[val_mask],
                            test_probs, test["label"].values[test_mask],
                            val_states=val_states, test_states=test_states,
                            suffix=f"_{args.symbol}")
        progress(90)

        # 融合预测落盘（供看板/回测优先使用；日期与序列窗口对齐）
        from models.fusion import StackingFusion
        fusion = StackingFusion().fit(val_probs, val["label"].values[val_mask])
        fused = fusion.predict_proba(test_probs)
        save_predictions(d_te, y_te, fused, c_te,
                         RESULTS_DIR / f"fusion_predictions_{args.symbol}.csv")
        # 注：fusion_metrics_{symbol}.json 已由 run_fusion_pipeline 落盘（含顶层 accuracy/auc）
    except Exception as e:
        print(f"训练失败: {e}", flush=True)
        set_status("error", 0, error=str(e))
        raise

    set_status("done", 100)
    print("\n全部完成，启动看板: streamlit run app/main.py")


if __name__ == "__main__":
    main()
