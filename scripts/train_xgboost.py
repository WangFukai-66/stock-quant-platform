"""训练 XGBoost 基线。

用法: python scripts/train_xgboost.py --symbol 600519
"""
import argparse

from models.xgboost_model import XGBBaseline
from models.common import RESULTS_DIR, save_json, save_predictions
from scripts.pipeline import prepare_tabular


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600519", help="股票代码")
    ap.add_argument("--start", default="20150101")
    args = ap.parse_args()

    train, val, test, _ = prepare_tabular(args.symbol, start=args.start)
    print(f"[{args.symbol}] train={len(train)} val={len(val)} test={len(test)}")

    model = XGBBaseline()
    model.fit(train, val)
    test_metrics = model._evaluate(test, "test")
    save_json(test_metrics, RESULTS_DIR / "xgb_metrics.json")

    imp = model.importance()
    imp.to_csv(RESULTS_DIR / "importance.csv", index=False)
    print(imp.to_string(index=False))

    prob = model.predict_proba(test)
    save_predictions(test["日期"].values, test["label"].values, prob.values,
                     test["收盘"].values, RESULTS_DIR / "xgb_predictions.csv")
    model.save()


if __name__ == "__main__":
    main()
