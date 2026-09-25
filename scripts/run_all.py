"""P1-10 一键全链路：数据采集 → 特征工程 → 模型训练 → 回测 → 预警。

用法:
    python scripts/run_all.py --symbol 600519 [--skip-train] [--epochs 30]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from backtest.engine import BacktestEngine, BacktestConfig
from data.fetcher import fetch_daily, fetch_news
from models.common import RESULTS_DIR, save_json
from scripts.pipeline import prepare_sequence, prepare_tabular

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def step(name: str):
    print(f"\n{'=' * 60}\n[步骤] {name}\n{'=' * 60}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600519")
    ap.add_argument("--start", default="20150101")
    ap.add_argument("--epochs", type=int, default=30, help="LSTM 训练轮数")
    ap.add_argument("--skip-train", action="store_true", help="跳过训练，用已有模型/预测")
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()

    # 1. 数据采集
    step(f"数据采集 {args.symbol}")
    ohlc = fetch_daily(args.symbol, adjust="qfq", start=args.start)
    news = fetch_news(args.symbol)
    print(f"日K: {len(ohlc)} 行 | 新闻: {len(news)} 条")

    # 2. 模型训练
    if not args.skip_train:
        step("XGBoost 基线训练")
        train, val, test, _ = prepare_tabular(args.symbol, start=args.start)
        from models.xgboost_model import XGBBaseline
        xgb = XGBBaseline()
        xgb.fit(train, val)
        test_metrics = xgb._evaluate(test, "test")
        save_json(test_metrics, RESULTS_DIR / "xgb_metrics.json")
        xgb.importance().to_csv(RESULTS_DIR / "importance.csv", index=False)
        from models.common import save_predictions
        prob = xgb.predict_proba(test)
        save_predictions(test["日期"].values, test["label"].values, prob.values,
                         test["收盘"].values, RESULTS_DIR / "xgb_predictions.csv")
        xgb.save()

        step("LSTM 训练")
        from models.lstm_model import predict_proba, train_lstm
        (x_tr, y_tr, _, _), (x_va, y_va, _, _), (x_te, y_te, d_te, c_te), _ = \
            prepare_sequence(args.symbol, start=args.start)
        model = train_lstm(x_tr, y_tr, x_va, y_va, epochs=args.epochs)
        l_prob = predict_proba(model, x_te)
        from models.common import eval_classification
        save_json(eval_classification(y_te, l_prob), RESULTS_DIR / "lstm_metrics.json")
        save_predictions(d_te, y_te, l_prob, c_te, RESULTS_DIR / "lstm_predictions.csv")
    else:
        step("跳过训练，使用已有预测文件")

    # 3. 回测（优先 XGBoost 预测）
    step("A股规则回测")
    import pandas as pd
    pred_path = RESULTS_DIR / "xgb_predictions.csv"
    if not pred_path.exists():
        print("缺少预测文件，请先训练（去掉 --skip-train）")
        return
    pred = pd.read_csv(pred_path, parse_dates=["日期"])
    merged = ohlc.merge(pred[["日期", "预测概率"]], on="日期", how="inner")
    result = BacktestEngine(BacktestConfig(threshold=args.threshold)).run(merged, merged["预测概率"])
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")
    save_json(result.metrics, RESULTS_DIR / "backtest_metrics.json")

    # 4. 预警
    step("智能预警")
    from alert.rules import latest_signals, position_advice
    advice = position_advice(ohlc)
    print(f"仓位建议: {advice['建议仓位']} — {advice['依据']}")
    sig = latest_signals(ohlc)
    if not sig.empty:
        print(sig.to_string(index=False))
    else:
        print("当日无触发信号")

    step("完成")
    print("启动看板: streamlit run app/main.py")


if __name__ == "__main__":
    main()
