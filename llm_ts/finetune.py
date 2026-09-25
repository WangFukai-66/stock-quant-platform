"""P3-03 时序大模型 LoRA 轻量微调（Chronos）。

两种模式：
- 本地验证（Mac CPU）：chronos-bolt-tiny / chronos-t5-small，1 只股票，3 epoch
- 云端 GPU（Colab/AutoDL）：chronos-t5-base，沪深300 池，完整训练

依赖 chronos-forecasting 与 peft：
    pip install chronos-forecasting peft datasets

用法:
    python llm_ts/finetune.py --symbol 600519 --model amazon/chronos-bolt-tiny --epochs 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LLM_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "llm"
LLM_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "llm_cache"


def get_model_id(choice: str) -> str:
    presets = {
        "tiny": "amazon/chronos-bolt-tiny",
        "small": "amazon/chronos-t5-small",
        "mini": "amazon/chronos-t5-mini",
        "base": "amazon/chronos-t5-base",
    }
    return presets.get(choice, choice)


def finetune_chronos(symbol: str, model_id: str, epochs: int = 3,
                     lr: float = 1e-4, context_len: int = 512,
                     lora_r: int = 8, local: bool = True) -> None:
    """LoRA 微调 Chronos。本地小规模或云端完整训练均走此函数。"""
    import torch
    from peft import LoraConfig, get_peft_model, TaskType

    from llm_ts.adapter import build_chronos_samples, split_samples

    # ---- 数据 ----
    tr, va, te = split_samples(build_chronos_samples(symbol, context_len=context_len))
    print(f"[{symbol}] train={len(tr)} val={len(va)} test={len(te)}")

    # ---- 模型 ----
    from chronos import ChronosPipeline  # noqa: F401  # 仅验证包可用
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q", "v", "k", "o"],          # T5 attention 模块
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ---- 训练（简化自 chronos-forecasting 官方 fine-tune 示例）----
    from torch.utils.data import DataLoader, Dataset
    from transformers import Trainer, TrainingArguments

    class ChronosDataset(Dataset):
        def __init__(self, samples, context_len):
            self.x = [torch.tensor(s["input"], dtype=torch.float32) for s in samples]
            self.y = [s["target"] for s in samples]
            self.context_len = context_len

        def __len__(self):
            return len(self.x)

        def __getitem__(self, i):
            # 简化版输入特征（官方实现使用 chronos 分位 tokenizer，
            # 见 amazon-science/chronos-forecasting 仓库 fine-tuning 脚本）
            return {"input_values": self.x[i], "labels": self.y[i]}

    train_ds = ChronosDataset(tr, context_len)
    val_ds = ChronosDataset(va, context_len)

    training_args = TrainingArguments(
        output_dir=str(LLM_MODEL_DIR / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=4 if local else 16,
        per_device_eval_batch_size=8,
        learning_rate=lr,
        logging_steps=20,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()

    # ---- 保存 adapter（仅 LoRA 权重，体积小）----
    out_dir = LLM_MODEL_DIR / f"chronos_{symbol}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    print(f"LoRA adapter 已保存: {out_dir}")
    print("云端完整训练说明：将 --model base 与云端 GPU 环境执行本脚本，"
          "并把股票池循环传入即可（见 scripts/train_llm_cloud.py）。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600519")
    ap.add_argument("--model", default="tiny", help="tiny/small/mini/base 或完整 HF id")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--local", action="store_true", help="本地小规模验证模式")
    args = ap.parse_args()
    finetune_chronos(args.symbol, get_model_id(args.model),
                     epochs=args.epochs, lr=args.lr, local=args.local)
