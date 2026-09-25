# A股量化回测与智能交易预警系统

基于多模型融合（LSTM + XGBoost + Transformer）的 A 股量化投资分析平台，包含数据采集、特征工程、模型预测、A股规则回测与智能交易预警全链路，前端为 Streamlit 六页签看板。

> 雏雁计划项目 | 国际学院 · 智能科学与技术专业

## 快速开始

```bash
# 1. 安装依赖（建议 Python 3.10+）
pip install -r requirements.txt

# 2. 拉取示例数据（贵州茅台）
python -m data.fetcher 600519

# 3. 启动看板
streamlit run app/main.py
```

浏览器打开 http://localhost:8501，在侧边栏输入股票代码即可浏览六大功能页签。

## 一键全链路

```bash
python scripts/run_all.py --symbol 600519
```

自动执行：数据采集 → 特征工程 → 模型训练（或加载）→ 回测 → 预警信号。

## 项目结构

```
├── data/          # akshare 数据采集（parquet 本地缓存）
├── factors/       # 技术指标（MA/MACD/RSI/KDJ/BOLL）与量价因子
├── models/        # XGBoost / LSTM / Transformer / Stacking 融合 / 动态权重
├── backtest/      # A股规则回测引擎（T+1、涨跌停、手续费、印花税、滑点）
├── alert/         # 规则预警引擎（买卖点、风险、仓位建议）
├── app/           # Streamlit 六页签看板
├── nlp/           # 中文舆情情绪分析（chinese-roberta-wwm-ext）
├── vision/        # K线形态 CNN 识别（ResNet18）
├── llm_ts/        # 时序大模型调研 / 数据适配 / LoRA 微调 / 评估
├── scripts/       # 训练与运行脚本
├── docs/          # 选型报告与论文大纲
└── results/       # 指标 JSON、图表、模型权重
```

## 文档

- [PLAN.md](PLAN.md) — 总执行计划
- [PROMPTS.md](PROMPTS.md) — vibe coding 提示词库
- [PRODUCT_VISION.md](PRODUCT_VISION.md) — 产品愿景

## 常见问题

- **akshare 接口超时**：数据已缓存到 `data/cache/`，再次运行直接读缓存；更换接口需改 `data/fetcher.py` 中的来源函数。
- **Mac CPU 训练慢**：模型规模已按 CPU 适配（序列 60、隐藏层 64、epoch 50）；若需更大规模，见 `llm_ts/` 云端 GPU 脚本。
- **情绪模型首次加载**：chinese-roberta-wwm-ext 首次运行会自动下载约 400MB 权重；无网络时自动降级为规则打分。
- **预测无模型文件**：先运行 `python scripts/train_all.py` 训练并保存模型，看板会自动加载。

## 风险提示

本项目仅用于学术研究与技术演示，不构成任何投资建议。A 股投资有风险，入市需谨慎。
