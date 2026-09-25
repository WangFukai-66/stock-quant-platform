"""P3-02 数据适配：A股日K → 时序大模型输入格式。

目标模型：Chronos（chronos-forecasting 包，量化 token 序列）。
上下文窗口默认 512 个交易日，预测 1 步（次日收盘）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.quote import get_history

LLM_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "llm_cache"
CONTEXT_LEN = 512
PRED_LEN = 1


def build_chronos_samples(symbol: str, context_len: int = CONTEXT_LEN,
                          start: str = "20150101") -> list[dict]:
    """收盘价序列 → 单变量预测样本。

    每个样本: {"input": 前 context_len 个收盘价, "target": 第 context_len+1 个收盘价,
                "date": 目标日}
    """
    ohlc = get_history(symbol, period="daily", start=start)
    close = ohlc["收盘"].astype(np.float64).values
    dates = pd.to_datetime(ohlc["日期"]).values
    samples = []
    for i in range(context_len, len(close)):
        samples.append({
            "input": close[i - context_len:i],
            "target": close[i],
            "date": dates[i],
        })
    return samples


def split_samples(samples: list[dict], train_ratio: float = 0.7, val_ratio: float = 0.15):
    """严格时序切分。"""
    n = len(samples)
    tr_end, va_end = int(n * train_ratio), int(n * (train_ratio + val_ratio))
    return samples[:tr_end], samples[tr_end:va_end], samples[va_end:]


def save_samples(samples: list[dict], path: Path) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"input": s["input"].tolist(), "target": float(s["target"]),
             "date": str(s["date"])} for s in samples]
    path.write_text(json.dumps(data), encoding="utf-8")
    print(f"样本已保存: {path} ({len(data)} 条)")


def load_samples(path: Path) -> list[dict]:
    import json
    data = json.loads(path.read_text(encoding="utf-8"))
    return [{"input": np.array(d["input"], dtype=np.float64),
             "target": d["target"], "date": d["date"]} for d in data]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600519")
    args = ap.parse_args()
    tr, va, te = split_samples(build_chronos_samples(args.symbol))
    LLM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_samples(tr, LLM_DATA_DIR / f"{args.symbol}_train.json")
    save_samples(va, LLM_DATA_DIR / f"{args.symbol}_val.json")
    save_samples(te, LLM_DATA_DIR / f"{args.symbol}_test.json")
    print(f"train={len(tr)} val={len(va)} test={len(te)}")
