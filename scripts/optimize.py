"""P2A-04 贝叶斯优化调参（optuna）：XGBoost + LSTM 超参数搜索，对比网格基线。

用法: python scripts/optimize.py --symbol 600519 --trials 20 [--model xgb|lstm]
"""
from __future__ import annotations

import argparse

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from models.common import RESULTS_DIR, save_json
from scripts.pipeline import prepare_sequence, prepare_tabular

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_xgb(symbol: str, start: str, n_trials: int) -> dict:
    import xgboost as xgb
    from factors.process import FACTOR_COLS
    train, val, test, _ = prepare_tabular(symbol, start=start)
    x_tr, y_tr = train[FACTOR_COLS], train["label"]
    x_va, y_va = val[FACTOR_COLS], val["label"]

    # 网格搜索基线
    baseline = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                 eval_metric="auc", tree_method="hist")
    baseline.fit(x_tr, y_tr)
    base_auc = roc_auc_score(y_va, baseline.predict_proba(x_va)[:, 1])
    print(f"[网格基线] val auc = {base_auc:.4f}")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "eval_metric": "auc", "tree_method": "hist", "random_state": 42,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(x_tr, y_tr)
        return roc_auc_score(y_va, model.predict_proba(x_va)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    result = {"model": "xgb", "grid_baseline_auc": round(base_auc, 4),
              "best_auc": round(study.best_value, 4),
              "best_params": study.best_params,
              "n_trials": n_trials}
    save_json(result, RESULTS_DIR / "optuna_best.json")
    print(f"[XGBoost] 网格基线 {base_auc:.4f} → optuna 最优 {study.best_value:.4f}")
    print(f"最优参数: {study.best_params}")
    return result


def optimize_lstm(symbol: str, start: str, n_trials: int) -> dict:
    from models.lstm_model import LSTMClassifier, predict_proba
    from models.common import get_device
    import torch

    (x_tr, y_tr, _, _), (x_va, y_va, _, _), _, _ = prepare_sequence(symbol, start=start)
    device = get_device()

    def objective(trial):
        hidden = trial.suggest_categorical("hidden", [32, 64, 128])
        layers = trial.suggest_categorical("layers", [1, 2])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        model = LSTMClassifier(x_tr.shape[2], hidden_size=hidden,
                               num_layers=layers, dropout=dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        xt = torch.tensor(x_tr, dtype=torch.float32, device=device)
        yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
        for _ in range(15):                       # 调参用短训练
            model.train()
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 64):
                idx = perm[i:i + 64]
                opt.zero_grad()
                loss = loss_fn(model(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        prob = predict_proba(model, x_va, device=device)
        return roc_auc_score(y_va, prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    result = {"model": "lstm", "grid_baseline_auc": "hidden=64,layers=2,lr=1e-3",
              "best_auc": round(study.best_value, 4),
              "best_params": study.best_params,
              "n_trials": n_trials}
    save_json(result, RESULTS_DIR / "optuna_best.json")
    print(f"[LSTM] optuna 最优 auc {study.best_value:.4f} 参数 {study.best_params}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600519")
    ap.add_argument("--start", default="20150101")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--model", default="xgb", choices=["xgb", "lstm"])
    args = ap.parse_args()

    if args.model == "xgb":
        optimize_xgb(args.symbol, args.start, args.trials)
    else:
        optimize_lstm(args.symbol, args.start, args.trials)
