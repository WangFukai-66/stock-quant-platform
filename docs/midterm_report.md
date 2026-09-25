# 中期研究报告要点（阶段2 里程碑）

> 对应计划「阶段2 里程碑：融合引擎与单模型回测对比，产出中期研究报告要点」。

## 一、阶段2 完成内容

### 2A 数值时序
- Transformer 时序模型（CLS + 位置编码，d_model=64/heads=4/layers=2，Mac CPU 可训练）
- Stacking 融合：LSTM + XGBoost + Transformer → LogisticRegression 元学习器
- 动态权重：市场状态四象限（高/低波动 × 趋势/震荡），每象限独立元学习器
- optuna 贝叶斯调参（XGBoost/LSTM），对比网格搜索基线

### 2B 文本
- 舆情情绪三分类：规则词典（默认）+ chinese-roberta-wwm-ext（深度，自动降级）
- 情绪因子：日情绪均值 / 5 日趋势 / 极端情绪占比 / 舆情热度
- 情绪因子增量检验（AUC 对比）+ 新闻事件研究（前后 60 分钟窗口）

### 2C 视觉
- 标准化 K 线渲染（64×64，无文字，含 MA5/MA20）
- 合成形态数据（头肩顶/双底/上升三角/箱体/突破 5 类）
- ResNet18 微调形态识别 + 形态概率作为第 4 路融合信号

## 二、关键实验结论（模板，训练后填写）

| 实验 | 结论来源 | 待填 |
|---|---|---|
| 融合 vs 单模型 AUC | results/fusion_compare.csv | 运行 scripts/train_all.py 后填写 |
| 动态权重 vs 静态 Stacking | results/fusion_metrics.json | 同上 |
| 情绪因子增量 | results/sentiment_factor_report.md | 运行情绪因子对比脚本 |
| 视觉信号增量 | results/vision_fusion_report.md | 运行 scripts/vision_fusion.py |
| optuna 最优参数 | results/optuna_best.json | 运行 scripts/optimize.py |

## 三、阶段3 展望（衔接）

- 时序大模型选型已定（Chronos，见 docs/llm_survey.md）
- 数据适配/微调/评估代码就绪（llm_ts/）
- 论文大纲已建（docs/paper_outline.md），阶段3 完成后填入对比实验即可成稿
