# P3-01 开源时序大模型选型报告

> 调研日期：2026-09。目标：为 A 股日频量价预测选择可微调的开源时序大模型。
> 硬约束：本机为 Mac CPU（小模型可本地推理/小规模微调），完整微调走云端 GPU（Colab/AutoDL）。

## 候选模型对比

| 维度 | Chronos(-Bolt) | TimesFM | Timer | Autoformer |
|---|---|---|---|---|
| 出品方 | Amazon | Google DeepMind | 蚂蚁集团(THU) | 清华 AutoML |
| 架构 | T5 encoder-decoder，量化 token | decoder-only，patch 编码 | decoder-only，因果 Transformer | 传统深度模型+自相关分解 |
| 最小参数量 | 8M(bolt-tiny)/20M(small)/46M(mini) | 200M(1.0) | 37M(16B 预训练小版本) | ~10M(可自训练) |
| 输入形态 | 数值序列 → 分位 token | 时序 patch 序列 | 序列 patch | 多变量序列 |
| 开源协议 | Apache 2.0 | Apache 2.0 | CC-BY-NC(学术) | MIT |
| HF 访问 | 公开，无需授权 | 需要接受 gated 条款并登录 | 公开 | 公开 |
| CPU 推理 | 小模型可行（秒级） | 200M 较慢 | 可行 | 快 |
| LoRA 微调支持 | 官方示例完善 | 社区支持 | 官方/社区均有 | 无（需全参） |
| 金融场景适配 | 通用预测，可零样本 | 通用预测 | 专为金融时序设计（A股预训练） | 通用 |
| 中文社区资料 | 较多 | 一般 | 多（国内） | 多 |

## 结论与推荐

**首选：Chronos（Amazon，Apache 2.0）**
- 理由：许可证最宽松（可商用/开源发布），HuggingFace 直接下载无需 gated 授权，
  官方提供完整的 LoRA 微调示例，8M/20M 小模型可在本机 Mac CPU 完成一轮微调验证。
- 版本：本地验证用 `amazon/chronos-bolt-tiny` 或 `amazon/chronos-t5-small`；
  云端 GPU 完整微调用 `amazon/chronos-t5-base`。

**备选：Timer（蚂蚁）**
- 理由：专为金融时序预训练（A股/美股），语义相关性更强；但许可证为 CC-BY-NC
  （不可商用），学术项目可用，开源发布需谨慎标注。

**不选：TimesFM** — gated 访问增加复现门槛；**Autoformer** — 非大模型路线，
与"时序大模型微调"主题不符，但可作为论文 baseline 对照。

## 迁移策略（Mac CPU → 云端 GPU）

1. 本地（Mac CPU）：chronos-bolt-tiny + 1 只股票 + 3 epoch 跑通 LoRA 管道，验证代码正确性；
2. 云端（Colab T4 / AutoDL A100）：chronos-t5-base + 沪深300 股票池 + 完整训练；
3. 代码统一 `device` 抽象（见 models/common.py），`llm_ts/finetune.py` 一个脚本双端可用。

## 风险

- Chronos 为通用时序模型，对 A 股低信噪比收益序列的零样本能力有限，
  微调后需与阶段2融合引擎同基准对比，不预设结论。
- 预测目标建议用「收盘价」或「对数收益」而非涨跌标签，Chronos 输出为数值分布。
