# PROMPTS — vibe coding 提示词库

> 用法：按「阶段-任务编号」逐条复制给 AI 执行（如"执行 P1-05"）。
> 每条包含【目标】【输入】【要求】【验收】。项目已存在的模块优先复用，不要重复造轮子。

---

## 阶段0：环境与骨架

### P0-01 项目骨架
【目标】初始化项目目录结构与依赖清单。
【输入】当前空项目目录。
【要求】1) 创建 data/factors/models/backtest/alert/app/scripts/results/nlp/vision/llm_ts/docs 目录及空 __init__.py；2) requirements.txt 含 akshare/pandas/numpy/streamlit/xgboost/torch/plotly/sqlalchemy/matplotlib/scikit-learn/optuna/transformers/datasets/peft；3) README.md 写明安装与启动方法。
【验收】`pip install -r requirements.txt` 通过，目录结构齐全。

### P0-02 数据采集模块
【目标】封装 akshare 数据源，实现行情/财务/新闻三类数据的统一采集与本地缓存。
【输入】requirements.txt 已装 akshare。
【要求】1) `data/fetcher.py` 提供 fetch_daily(symbol)、fetch_minute(symbol)、fetch_financial(symbol)、fetch_news(symbol) 四个函数；2) 统一 DataFrame 中文列名：日期/开盘/收盘/最高/最低/成交量；3) parquet 缓存到 data/cache/，7 天内不重复请求；4) 请求间隔 ≥1s 并带重试 3 次；5) 异常时返回空 DataFrame 并记录 warning。
【验收】`python -m data.fetcher 600519` 输出四类数据摘要并落盘缓存。

---

## 阶段1：Prototype 原型产品

### P1-01 行情模块
【目标】日K/分钟线数据服务与前复权处理。
【输入】data/fetcher.py。
【要求】1) 支持 qfq 前复权模式；2) 分钟线聚合为 5/15/30/60 分钟可选；3) 提供 `data/quote.py` 的 get_history(symbol, period, adjust) 统一接口；4) 计算涨跌幅、换手率衍生列。
【验收】取平安银行日K 250 条，含前复权价格与涨跌幅。

### P1-02 财报模块
【目标】三大报表摘要与核心财务指标计算。
【输入】data/fetcher.py 财务接口。
【要求】1) 提取资产负债表/利润表/现金流量表关键科目；2) 计算 ROE、毛利率、净利率、资产负债率、经营现金流/净利润；3) 输出近 5 年指标时间序列；4) 数据缺失用 NaN 保留不伪造。
【验收】输出贵州茅台近 5 年 ROE/毛利率表格，缺失值可见。

### P1-03 舆情模块
【目标】个股新闻采集与基础情感打分。
【输入】data/fetcher.py 新闻接口。
【要求】1) 新闻去重（标题+日期）；2) 规则情感打分：正负面词典匹配输出 [-1,1] 分与三分类；3) 每日舆情热度 = 当日新闻数；4) 输出舆情时间轴 DataFrame。
【验收】输出个股近 30 天新闻数、情绪分、热度折线数据。

### P1-04 特征工程
【目标】技术指标与量价因子计算、标准化。
【输入】日K数据。
【要求】1) `factors/indicators.py` 实现 MA5/10/20/60、MACD、RSI、KDJ、BOLL、ATR；2) 量价因子：量比、5日/20日动量、波动率、振幅；3) `factors/process.py` 提供标准化（z-score）与标签生成（次日涨跌、次日收益率）；4) 所有指标只用历史数据（防未来函数）。
【验收】输入日K输出 ≥20 列因子表，无 NaN 行（首 60 行可截断）。

### P1-05 XGBoost 基线
【目标】XGBoost 次日涨跌预测基线。
【输入】factors 因子表；股票池：沪深300 成分。
【要求】1) 滚动时间切分（train/val/test 按时间，无 shuffle）；2) 输出 accuracy/AUC/混淆矩阵；3) 特征重要性 Top20 存 results/；4) 预测概率落盘供回测调用；5) 参数规模适合 Mac CPU（n_estimators≤200, max_depth≤6）。
【验收】运行 scripts/train_xgboost.py 生成 results/xgb_metrics.json 与 importance.csv。

### P1-06 LSTM 预测
【目标】LSTM 滑窗序列预测次日涨跌。
【输入】标准化因子表。
【要求】1) 滑窗 60 天；2) 隐藏层 ≤64，epoch ≤50，batch 32，PyTorch CPU；3) 严格时序切分，保存验证集 loss 曲线；4) 输出预测概率与 test 集 AUC；5) 模型保存 models/lstm.pt。
【验收】运行 scripts/train_lstm.py 生成 results/lstm_metrics.json。

### P1-07 回测引擎
【目标】严格模拟 A 股交易规则的回测引擎。
【输入】日K + 预测信号（0/1 或概率阈值）。
【要求】1) `backtest/engine.py`：T+1（当日买入次日可卖）、涨跌停价不可成交、手续费万 2.5 双边、印花税卖出千 1、滑点 1 分钱；2) 初始资金 100 万，全仓进出；3) 输出总收益、年化、夏普、最大回撤、胜率、交易次数；4) 生成资金曲线与回撤序列；5) 信号只能用当日收盘后信息。
【验收】运行 scripts/run_backtest.py 输出指标 JSON 与资金曲线图。

### P1-08 预警模块
【目标】规则引擎实时预警信号。
【输入】日K + 因子表。
【要求】1) `alert/rules.py`：金叉/死叉、涨跌停触板、放量异动（量比>2）、MACD 顶背离/底背离、回撤阈值；2) 每条信号含类型/方向/触发价/说明；3) 仓位建议：强信号满仓、弱信号半仓、风险信号降仓；4) 输出当日信号列表。
【验收】输入个股数据输出信号清单，每条可解释。

### P1-09 Streamlit 看板
【目标】六页签可视化看板。
【输入】上述所有模块。
【要求】1) `app/main.py` 侧边栏股票代码/周期/日期范围选择；2) 六页签：首页(概览)/行情/财报/舆情/预测/回测/预警；3) plotly 交互图（K线+成交量+指标副图）；4) 预测页展示 XGBoost/LSTM 概率；5) 预警页表格+高亮；6) 页面加载用缓存装饰器。
【验收】`streamlit run app/main.py` 六页签均可交互展示。

### P1-10 集成联调
【目标】全链路一键运行与文档。
【输入】全部模块。
【要求】1) `scripts/run_all.py` 一键：采集→因子→训练（或加载已训练模型）→回测→预警；2) README 使用说明（含常见报错处理）；3) 演示脚本按顺序输出各阶段关键结果。
【验收】新环境按 README 可复现全链路。

---

## 阶段2：三向深耕

### P2A-01 Transformer 时序模型
【目标】纯 PyTorch Transformer 次日涨跌预测。
【输入】滑窗因子序列（同 LSTM）。
【要求】1) Positional Encoding + 2 层 Multi-Head Attention（d_model=64, heads=4）；2) CLS token 输出接分类头；3) CPU 可训练（epoch≤50）；4) 与 LSTM 同样严格时序切分。
【验收】scripts/train_transformer.py 输出 AUC ≥ LSTM 基线或报告差距原因。

### P2A-02 Stacking 融合引擎
【目标】LSTM + XGBoost + Transformer 三模型 Stacking。
【输入】三基模型验证集预测概率。
【要求】1) `models/fusion.py`：基模型输出作为特征，元学习器 LogisticRegression/MLP 训练；2) 严格用验证集预测训练元学习器，测试集评估（防泄露）；3) 对比单模型与等权平均的 AUC/准确率；4) 输出各模型权重/重要性。
【验收】融合 AUC 不低于最佳单模型，产出对比表存 results/fusion_metrics.json。

### P2A-03 动态权重机制
【目标】市场状态自适应融合权重。
【输入】融合引擎 + 市场状态特征（20 日波动率分位、趋势方向）。
【要求】1) 将市场划分为 高波动/低波动 × 趋势/震荡 四象限；2) 每象限单独学习最优权重；3) 预测时按当前状态选用权重；4) 回测对比静态权重。
【验收】动态权重在分段回测中夏普或胜率优于静态权重。

### P2A-04 贝叶斯优化调参
【目标】optuna 自动调参，对比网格搜索。
【输入】基模型训练流程。
【要求】1) 对 XGBoost/LSTM 各运行 ≥20 次试验；2) 目标 = 验证集 AUC；3) 输出 best params 与参数重要性；4) 与手动参数/网格搜索结果对比表。
【验收】scripts/optimize.py 生成 results/optuna_best.json 与对比表。

### P2B-01 舆情情绪分类
【目标】中文预训练模型情绪三分类。
【输入】个股新闻文本（或开源金融情绪数据集）。
【要求】1) `nlp/sentiment.py` 加载 chinese-roberta-wwm-ext，输出 正面/中性/负面 概率；2) CPU 批量推理，batch=8；3) 无 GPU 时提供规则打分降级路径；4) 结果缓存避免重复推理。
【验收】对 20 条样例新闻输出情绪概率，方向与人工判断一致。

### P2B-02 情绪因子回测
【目标】情绪分作为特征/信号验证增量收益。
【输入】P2B-01 情绪序列 + 因子表。
【要求】1) 构建情绪因子：日情绪均值、5 日趋势、极端情绪占比；2) 对比 加入/不加入情绪因子的 XGBoost 与策略回测；3) 输出增量收益结论。
【验收】results/sentiment_factor_report.md 说明情绪因子是否带来增量及显著性。

### P2B-03 新闻事件驱动
【目标】突发新闻 → 个股异常反应检测。
【输入】新闻流 + 分钟线。
【要求】1) 新闻发布前后 60 分钟收益与成交量偏离检测；2) 事件窗口异常收益率统计（事件研究法）；3) 输出高影响新闻清单。
【验收】对样例新闻输出事件窗口收益柱状图与显著性标记。

### P2C-01 K线图渲染
【目标】标准化 K 线图数据集生成。
【输入】日K数据。
【要求】1) `vision/render.py`：matplotlib 渲染 64×64 RGB 灰度统一背景，含 MA5/MA20；2) 无坐标轴文字（防模型依赖文字）；3) 支持批量导出。
【验收】生成 100 张形态样例图，肉眼可辨形态。

### P2C-02 CNN 形态识别
【目标】ResNet18 微调识别 K 线形态。
【输入】P2C-01 图像 + 形态标签（头肩顶/双底/上升三角/突破等 5 类）。
【要求】1) torchvision ResNet18 改输出 5 类；2) 合成形态 + 人工标注混合训练集；3) 数据增强（翻转/对比度）；4) CPU 训练 5-10 epoch，输出分类报告。
【验收】测试集准确率 ≥60%，输出混淆矩阵图。

### P2C-03 视觉信号融合
【目标】CNN 形态概率作为第 4 路融合信号。
【输入】CNN 模型 + 融合引擎。
【要求】1) 对预测日近 60 日窗口渲染 K 线图 → CNN 输出形态概率向量；2) 形态概率拼入 Stacking 元特征；3) 回测对比 三模型融合 vs 四路信号融合。
【验收】results/vision_fusion_report.md 输出增量结论。

---

## 阶段3：时序大模型微调

### P3-01 调研选型
【目标】开源时序大模型对比选型。
【输入】公开资料（Chronos/TimesFM/Timer/Autoformer）。
【要求】1) 对比：参数量、输入形态、开源协议、CPU 推理可行性、中文社区支持；2) 结合本机 Mac CPU 给出推荐（默认 Chronos-t5-small 或 TimesFM）；3) 输出 docs/llm_survey.md。
【验收】选型报告含对比表与明确推荐及理由。

### P3-02 数据适配
【目标】A股量价数据转换为目标模型输入格式。
【输入】日K + 选型结论。
【要求】1) `llm_ts/adapter.py`：收盘价序列 → 模型输入张量/上下文窗口；2) 构造预测任务样本（上下文 512 点，预测 1 步）；3) 划分 train/val/test 并落盘。
【验收】生成适配后的训练样本文件，形状打印正确。

### P3-03 微调实验
【目标】LoRA 轻量微调。
【输入】适配样本。
【要求】1) `llm_ts/finetune.py`：peft LoRA（r=8）冻结主干；2) 优先本地小规模验证（1 只股票、少 epoch），给出云端 GPU 完整训练脚本与说明；3) 记录训练 loss。
【验收】本地能完成一轮微调并保存 adapter 权重；云训练脚本可直接运行。

### P3-04 对比评估
【目标】微调模型 vs 融合引擎统一基准回测。
【输入】微调模型预测 + 融合引擎预测。
【要求】1) 同一 test 区间、同一回测引擎、同一指标口径；2) 输出 MAE/MSE/方向准确率 + 回测收益对比；3) 显著性检验（Diebold-Mariano 或配对检验）。
【验收】results/llm_vs_fusion.md 对比表与结论。

### P3-05 学术产出
【目标】论文/技术白皮书结构与开源整理。
【输入】全部实验产物。
【要求】1) docs/paper_outline.md：摘要/引言/数据/方法（融合引擎+三向深耕）/实验/结论；2) 关键图表清单与来源脚本；3) 代码 README 开源说明（许可、复现步骤）。
【验收】论文大纲可直接扩充成稿，复现步骤可操作。
