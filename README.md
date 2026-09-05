# CycAMP-FusionDemo

本科科研训练项目：面向环肽的抗菌活性预测研究 Demo。项目比较序列理化特征、冻结 ESM-2 表示、LoRA 迁移学习和预测结构模型衍生描述符，并提供 Streamlit 交互界面。

> 本项目仅用于科研训练与计算筛选。页面输出均为模型预测得分，不是校准概率或实验结论；候选肽的真实抗菌作用仍须通过体外实验验证。

## 最终结果

核心环肽数据集包含 284 个化学实体（214 个活性、70 个非活性）；通用 AMP 迁移集包含 7,500 条序列。评价采用按环肽分组的三折外层交叉验证和 1,000 次组级 bootstrap，主指标为 MCC。

| 评价范围 | 特征/模型 | OOF MCC | F1 | ROC-AUC | PR-AUC | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 全核心集（284 条） | M0：人工特征，嵌套选择 | 0.3272 | 0.8652 | 0.7817 | 0.9192 | 0.7817 |
| 全核心集（284 条） | M1：冻结 ESM-2，嵌套选择 | 0.2632 | 0.7961 | 0.7045 | 0.8697 | 0.7042 |
| HighFold2 共同子集（277 条） | M0 + M1，嵌套选择 | 0.4021 | 0.8651 | 0.7970 | 0.9202 | 0.7906 |
| HighFold2 共同子集（277 条） | M2：人工特征 + ESM-2 + 结构描述符，嵌套选择 | 0.3313 | 0.8451 | 0.7824 | 0.9197 | 0.7617 |

在同一 HighFold2 共同子集上，加入当前结构描述符后未观察到优于 M0 + M1 的 MCC。这是有效的探索性结果，而非结构信息无效的普遍结论。

## 方法概览

- **M0**：长度、电荷、疏水性、氨基酸组成、成环信息和环形邻接特征；比较 LR、RBF-SVM、RF 与 HistGradientBoosting。
- **M1**：冻结 `facebook/esm2_t12_35M_UR50D` 表示；头尾成环肽对循环移位序列的嵌入求平均。
- **M1-LoRA**：在通用 AMP 与环肽数据上进行两阶段迁移，LoRA 仅作用于注意力 query/value。
- **M2**：融合 M0 特征、ESM-2 表示和 HighFold2 预测结构模型衍生的 10 项三维描述符。

HighFold2 结构仅称为“预测结构模型”，RDKit 结果仅称为“三维候选构象”；两者均不代表实验测定结构或已验证活性构象。

## 运行 Demo

推荐使用 Conda 和 Python 3.11：

```bash
conda env create --file environment.yml
conda activate cycamp-fusion-demo
streamlit run app.py
```

Demo 接受 5–50 位标准氨基酸单字母序列，并支持头尾成环、二硫键等成环类型。

出于数据使用条款、文件体积与可追溯性考虑，原始数据、模型权重、嵌入缓存、批量构象和报告二进制均未随仓库发布。因此，克隆后的项目可查看代码和界面逻辑；要得到模型预测得分，需要依照相应数据来源与许可条件准备模型工件。

## 代码结构

```text
app.py                 Streamlit 交互 Demo
src/cycamp/            数据处理、特征、模型、结构和推理实现
scripts/               数据处理、训练、结构导入和环境辅助脚本
configs/demo.yaml      统一配置
tests/                 单元测试与应用冒烟测试
environment.yml        Conda 环境定义
```

## 复现与限制

- MIC 以 25 µg/mL 作为操作性二分类阈值，不能解释为临床折点。
- 数据库记录、模型权重和上游软件须按各自许可与引用要求获取。
- 核心集的序列家族独立性仍有限，现有指标可能高估面对全新家族的泛化能力。
- 训练时的 HighFold2 结构描述符与 Demo 运行时生成的 RDKit 三维候选构象来自不同结构来源，应谨慎解释。

## 致谢与上游项目

本项目使用或参考了 [ESM-2](https://github.com/facebookresearch/esm)、[Transformers](https://github.com/huggingface/transformers)、[PEFT](https://github.com/huggingface/peft)、[scikit-learn](https://github.com/scikit-learn/scikit-learn)、[RDKit](https://github.com/rdkit/rdkit)、[Streamlit](https://github.com/streamlit/streamlit) 与 [HighFold2](https://github.com/hongliangduan/HighFold2)。使用相关代码和数据前，请遵循其许可证、数据库条款及原始文献引用要求。
