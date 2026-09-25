"""训练 LSTM。

用法: python scripts/train_lstm.py --symbol 600519 --epochs 50
"""
import argparse

from models.common import RESULTS_DIR, save_json, save_predictions
from models.lstm_model import predict_proba, train_lstm
from scripts.pipeline import prepare_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600519", help="股票代码")
    ap.add_argument("--start", default="20150101")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--window", type=int, default=60)
    args = ap.parse_args()

    (x_tr, y_tr, _, _), (x_va, y_va, _, _), (x_te, y_te, d_te, c_te), _ = \
        prepare_sequence(args.symbol, window=args.window, start=args.start)
    print(f"[{args.symbol}] train={len(x_tr)} val={len(x_va)} test={len(x_te)}")

    model = train_lstm(x_tr, y_tr, x_va, y_va, epochs=args.epochs)
    prob = predict_proba(model, x_te)

    from models.common import eval_classification
    metrics = eval_classification(y_te, prob)
    print(f"[LSTM test] accuracy={metrics['accuracy']} auc={metrics['auc']}")
    save_json(metrics, RESULTS_DIR / "lstm_metrics.json")
    save_predictions(d_te, y_te, prob, c_te, RESULTS_DIR / "lstm_predictions.csv")


if __name__ == "__main__":
    main()
