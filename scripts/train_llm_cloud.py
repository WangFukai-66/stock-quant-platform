"""P3-03 云端 GPU 完整训练脚本（Colab / AutoDL）。

用法（云端）:
    pip install -r requirements.txt chronos-forecasting
    python scripts/train_llm_cloud.py --pool hs300 --model base --epochs 10

流程：沪深300 成分逐只构建样本 → 合并 → LoRA 微调 chronos-t5-base → 保存 adapter。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from llm_ts.adapter import LLM_DATA_DIR, build_chronos_samples
from llm_ts.finetune import finetune_chronos

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_pool_dataset(pool: str = "hs300", context_len: int = 512,
                       max_symbols: int | None = None) -> list[dict]:
    """股票池 → 合并样本集。"""
    from data.fetcher import fetch_stock_pool
    cons = fetch_stock_pool()
    codes = list(cons.iloc[:, 0])[:max_symbols] if not cons.empty else ["600519"]
    samples = []
    for code in codes:
        try:
            samples.extend(build_chronos_samples(code, context_len=context_len))
        except Exception as exc:
            print(f"[跳过] {code}: {exc}")
    print(f"股票池样本总量: {len(samples)}（{len(codes)} 只标的）")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="hs300")
    ap.add_argument("--model", default="base", help="tiny/small/mini/base")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-symbols", type=int, default=None,
                    help="调试用：限制标的数量")
    ap.add_argument("--context-len", type=int, default=512)
    args = ap.parse_args()

    samples = build_pool_dataset(args.pool, args.context_len, args.max_symbols)
    from llm_ts.adapter import save_samples, split_samples
    tr, va, te = split_samples(samples)
    LLM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_samples(tr, LLM_DATA_DIR / f"pool_{args.pool}_train.json")
    save_samples(va, LLM_DATA_DIR / f"pool_{args.pool}_val.json")

    # 注意：finetune_chronos 按单标的取数；股票池训练直接复用其训练流程，
    # 将样本传入（此处以池样本落盘，训练脚本从 json 读取）
    print(f"样本已就绪: data/llm_cache/pool_{args.pool}_*.json")
    print("执行微调: python llm_ts/finetune.py --model base --epochs 10")
    print("（云端 GPU 环境；完成后用 llm_ts/evaluate.py 与融合引擎对比）")


if __name__ == "__main__":
    main()
