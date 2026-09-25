"""数据准备公共流程：行情 → 因子表 → 时间切分 → 标准化 → 滑窗。"""
from __future__ import annotations

import argparse

import pandas as pd

from data.quote import get_history
from factors.process import (build_dataset, make_windows, split_temporal,
                             standardize)


def prepare_tabular(symbol: str, start: str = "20150101"):
    """表格型数据（XGBoost 用）：返回 train/val/test 及标准化参数。"""
    ohlc = get_history(symbol, period="daily", start=start)
    ds = build_dataset(ohlc)
    train, val, test = split_temporal(ds)
    train, val, test, stats = standardize(train, val, test)
    return train, val, test, stats


def prepare_sequence(symbol: str, window: int = 60, start: str = "20150101"):
    """序列型数据（LSTM/Transformer 用）：滑窗数组 + 时间切分。"""
    ohlc = get_history(symbol, period="daily", start=start)
    ds = build_dataset(ohlc)
    train, val, test = split_temporal(ds)
    train, val, test, stats = standardize(train, val, test)
    x_train, y_train, d_train, c_train = make_windows(train, window)
    x_val, y_val, d_val, c_val = make_windows(val, window)
    x_test, y_test, d_test, c_test = make_windows(test, window)
    return ((x_train, y_train, d_train, c_train),
            (x_val, y_val, d_val, c_val),
            (x_test, y_test, d_test, c_test), stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600519")
    args = ap.parse_args()
    tr, va, te, _ = prepare_tabular(args.symbol)
    print(f"train={len(tr)} val={len(va)} test={len(te)}")
    print(tr[["日期", "label"]].head())
