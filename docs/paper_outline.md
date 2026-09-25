# 论文 / 技术白皮书大纲

> 对应 PPT「预期成果：学术论文与研究报告」。
> 本文档为结构化大纲，各章节对应代码产物可直接引用。

## 标题（候选）

基于多模型融合与多源数据的 A 股量化预测与智能预警系统

## 摘要

- 背景：A 股高噪声、政策市、散户结构下的预测难题
- 方法：多源数据（行情/财报/舆情/视觉）→ 三基模型（XGBoost/LSTM/Transformer）→
  Stacking 融合 + 市场状态动态权重 → 时序大模型 LoRA 微调对照
- 结果：融合引擎 vs 单模型 AUC/回测指标对比；情绪因子、视觉信号增量检验
- 结论与展望

## 1 引言

1.1 研究背景与意义（PPT 项目愿景）
1.2 相关工作：机器学习量化（XGBoost/LSTM/Transformer）、金融情绪分析、
K线形态识别、时序大模型（Chronos/TimesFM/Timer）
1.3 本文贡献：
- 三模型 Stacking + 市场状态自适应动态权重
- 文本/视觉信号与数值时序的统一融合框架
- 时序大模型微调与轻量融合引擎的公平对比

## 2 数据与特征工程

2.1 数据源与采集（akshare；行情/财报/新闻三源；parquet 缓存架构）
2.2 特征工程：技术指标（MA/MACD/RSI/KDJ/BOLL/ATR）与量价因子
2.3 情绪因子构建（规则词典 + RoBERTa 三分类）
2.4 视觉表征（64×64 K线图 + 合成形态数据）
2.5 防未来函数与严格时序切分设计

## 3 方法

3.1 基模型：XGBoost / LSTM / Transformer（结构图与超参表）
3.2 Stacking 融合与元学习器
3.3 市场状态分域与动态权重机制（四象限）
3.4 情绪分析模型（chinese-roberta-wwm-ext）
3.5 K线形态识别（ResNet18 微调）
3.6 时序大模型微调（Chronos LoRA；选型论证见 docs/llm_survey.md）

## 4 回测与评估框架

4.1 A 股规则回测引擎（T+1、涨跌停、手续费、印花税、滑点）
4.2 评估指标：AUC/准确率/夏普/最大回撤/胜率/年化
4.3 样本外验证与牛/熊/震荡分段测试

## 5 实验

5.1 单模型对比（表：results/fusion_compare.csv）
5.2 融合引擎消融（Stacking vs 等权 vs 动态权重）
5.3 情绪因子增量检验（results/sentiment_factor_report.md）
5.4 视觉信号融合（results/vision_fusion_report.md）
5.5 时序大模型 vs 融合引擎（results/llm_vs_fusion.md，DM 检验）
5.6 回测结果与基准对比（results/backtest_metrics.json）

## 6 系统实现

6.1 模块化架构（六层解耦）
6.2 Streamlit 交互看板（六页签截图）
6.3 预警引擎与仓位建议

## 7 结论与展望

- 结论：融合有效性边界、情绪/视觉信号增量、大模型微调成本收益
- 局限：单标的回测、合成形态域差异、Mac CPU 规模限制
- 展望：多标的组合、分钟级、强化学习仓位、更大时序模型

## 关键图表清单

| 图表 | 来源脚本 |
|---|---|
| 系统架构图 | 手绘/绘图工具，参考 PLAN.md 目录结构 |
| K线+指标看板截图 | app/main.py 行情页 |
| 因子重要性 Top20 | scripts/train_xgboost.py → results/importance.csv |
| 模型对比表 | scripts/train_all.py → results/fusion_compare.csv |
| 资金曲线与回撤 | scripts/run_all.py 回测部分 |
| CNN 混淆矩阵 | vision/cnn_pattern.py → results/pattern_cnn_metrics.json |
| 事件研究窗口图 | nlp/sentiment_factor.py event_study |

## 开源发布清单

- 复现步骤：README.md（安装 → 数据 → 训练 → 看板）
- 许可：Apache 2.0（Chronos 相关遵循其协议；Timer 仅学术使用）
- 数据说明：akshare 公开数据，缓存文件不入库
