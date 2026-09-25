"""P1-05 XGBoost 次日涨跌预测基线。

用法:
    from models.xgboost_model import XGBBaseline
    m = XGBBaseline()
    m.fit(train, val)          # DataFrame，含 FACTOR_COLS 与 label
    prob = m.predict_proba(test)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import xgboost as xgb

from factors.process import FACTOR_COLS
from models.common import MODELS_DIR, eval_classification, save_json

DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cpu",
    "random_state": 42,
}


class XGBBaseline:
    def __init__(self, params: dict | None = None, model_path: Path | None = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model_path = model_path or MODELS_DIR / "xgb_model.json"
        self.model: xgb.XGBClassifier | None = None
        self.feature_names: list[str] = list(FACTOR_COLS)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None,
            early_stopping: int = 20, verbose: bool = True) -> dict:
        x_tr, y_tr = train[self.feature_names], train["label"]
        evals = None
        if val is not None and len(val) > 0:
            evals = [(x_tr, y_tr), (val[self.feature_names], val["label"])]
        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(x_tr, y_tr, eval_set=evals, verbose=verbose)
        return self._evaluate(train, "train") if evals is None else self._evaluate(val, "val")

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict_proba(df[self.feature_names])[:, 1], index=df.index)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict(df[self.feature_names]), index=df.index)

    def _evaluate(self, df: pd.DataFrame, name: str) -> dict:
        prob = self.predict_proba(df)
        m = eval_classification(df["label"].values, prob.values)
        print(f"[XGB {name}] accuracy={m['accuracy']} auc={m['auc']}")
        return m

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(self.model_path))

    def load(self) -> "XGBBaseline":
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(self.model_path))
        return self

    def importance(self, top: int = 20) -> pd.DataFrame:
        imp = pd.DataFrame({"因子": self.feature_names,
                            "重要性": self.model.feature_importances_})
        return imp.sort_values("重要性", ascending=False).head(top)


def train_xgboost(symbol: str, df: pd.DataFrame, train, val, test,
                  save_dir: Path | None = None) -> dict:
    """完整训练流程：训练 → 评估 → 特征重要性 → 落盘。"""
    from models.common import RESULTS_DIR, save_predictions

    m = XGBBaseline()
    m.fit(train, val)
    test_metrics = m._evaluate(test, "test")
    save_json(test_metrics, (save_dir or RESULTS_DIR) / "xgb_metrics.json")

    imp = m.importance()
    imp.to_csv((save_dir or RESULTS_DIR) / "importance.csv", index=False)
    print(imp.to_string(index=False))

    prob = m.predict_proba(test)
    save_predictions(test["日期"].values, test["label"].values, prob.values,
                     test["收盘"].values, (save_dir or RESULTS_DIR) / "xgb_predictions.csv")
    m.save()
    return test_metrics
