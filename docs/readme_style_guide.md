# README 排版准则

本准则由本仓库 `README.md` 的现有格式整理而来，用于其它项目按同一结构编写 README，保证跨项目的
文档骨架、命令写法和验收口径一致。

## 1. 总体原则

- 中文叙述，英文术语（`QAT`、`QuantONNX`、`smoke` 等）保留原文。
- 命令、参数、文件名、路径、指标名、数据类型一律使用反引号，例如 `--profile`、`config_siluInU8_attnS8_clsU16.json`、`mAP50-95`、`U16`。
- 每个步骤先给一句说明，紧跟可直接复制执行的命令；结论在命令之前，细节链接到专门文档。
- 任务小节顺序固定为：**检测 → 分割 → OBB → Pose → 分类 → 多 GPU**；同一任务在训练、评估、导出、图片验证各步骤中保持相同顺序。
- 已验证与未验证范围必须显式区分，未调优实验用 `⚠️` 警告块标注，不能与交付基线混排。
- README 只保留稳定结论与入口命令，实验细节、配置枚举、节点发现流程链接到专门文档。

## 2. 文档骨架

按以下顺序组织，章节标题使用对应级别：

```text
# 项目名
简介段（上游基础 + 能力范围 + 链接）
操作建议段（端到端最短路径）

## 精度结果
### 检测
### 分割
（按任务扩展）

## 配置章节（如 QAT 量化配置）
## 快速开始
### 1. 安装依赖
### 2. 准备数据集
### 3. 运行 Smoke
### 4. 正式训练
### 5. 评估精度
### 6. 导出 QuantONNX
### 7. 验收 Slim ONNX
### 8. 图片验证
## 模型部署
```

信息梯度为：**结果 → 配置 → 操作步骤 → 部署**，篇幅较长的操作流程放在文档后半部分。

## 3. 章节写法

### 3.1 标题与简介

- H1 使用仓库或项目名，不加副标题。
- 简介段包含三要素：上游基础（带链接与版本）、本仓库能力范围、目标产物。
- 操作建议段给出一条端到端路径：小数据集 1 epoch smoke → 评估 → 导出 → 板端部署 → 全量训练。

```markdown
# QAT.Ultralytics

本仓库基于 [Ultralytics-8.4](https://github.com/ultralytics/ultralytics/tree/v8.4.21)，提供
`YOLO26` 和 `YOLO11` 系列 PT2E QAT 的训练、QuantONNX 导出和 AXERA 部署兼容处理。

操作建议：初期请选用小批量数据集，仅训练 1 个 epoch；随后依次执行 eval.py 与 export.py 完成验证及模型导出，最后进行板端部署。
```

### 3.2 精度结果

- 每个任务一个 H3；小节以一句话说明指标口径与测试环境。
- 表格列固定为：模型 → 模式 → 配置 → 基线指标 → 量化指标 → 误差 → 耗时，数值列用 `---:` 右对齐。
- 表格下方用 `注：` 解释术语与统计口径；空值统一写 `-`，不写 `N/A`。
- 术语如 `end2end`、`clsU16`、`Speed(ms)` 必须在小节内解释。

```markdown
### 检测

以下为当前量化精度结果；`Speed(ms)` 为已完成 AXERA NPU 部署模型的板端实测耗时：

| 模型 | `end2end` | 配置 | FP32 mAP50-95 | FP32 mAP50 | 量化 mAP50-95 | 量化 mAP50 | Err mAP:50~95 | Err mAP:50 | Speed(ms) |
|---|---|---|---|---|---|---|---|---|---|
| YOLO26n | `true` | `config_siluInU16_attnS8_clsU16.json` | 40.24 | 55.79 | 39.61 | 55.63 | -0.63 | -0.16 | 3.761 |

注：`end2end=true` 为 `one2one` 模型，后处理无 `nms`；`end2end=false` 为 `one2many` 模型，后处理需 `nms`。
```

### 3.3 配置章节

- 不展开配置全集，给“已验证组合 + 适用范围 + 重新生成节点的入口链接”。
- 明确哪些模型可使用快捷 profile（如 `--profile accuracy|throughput` 仅对应 YOLO26n），其余必须显式传入配置。
- 链接专门文档与 skill 时使用相对路径和 `$skill-*` 前缀。

### 3.4 快速开始

- 使用 `### 1.` 到 `### 8.` 的连续编号；不跳号、不合并步骤。
- 每个编号步骤内按任务拆 H4 小节，如 `#### 检测`、`#### 分割`。
- 每个任务小节采用三段式：**说明段 → 命令块 → 要点列表**。
- 一个步骤内含多个相关命令时，在同一个代码块内用 `# 注释` 区分，避免为每条命令单独建块。

````markdown
#### 检测

```bash
# 浮点基线
env PYTHONPATH="$PWD" \
  python eval.py float \
  --task detect --weights weights/yolo26n.pt \
  --data coco.yaml --device cuda:0 --imgsz 640
```
````

### 3.5 要点列表

任务小节的要点使用固定前缀，便于跨项目检索：

- `- **输出契约**：...` 说明部署输出张量。
- `- **Smoke 指标**：...` 说明链路验收数据（同时声明数据规模小、仅供链路验收）。
- `- **已支持**：...` 列出已接入的评估、导出、部署能力。
- `- **未接入**：...` 列出未完成项。
- `- **⚠️ 仅链路验收，精度未调优**：...` 标注未达交付标准的配置。

### 3.6 警告块

用 blockquote 加粗标题强调范围边界，正文可跟编号列表：

```markdown
> **⚠️ 精度验证范围说明**
>
> **检测**已完成正式精度实验，可作为交付基线。
>
> **OBB、Pose 和分类**目前仅完成主机侧链路 smoke，**未经正式精度调优**。使用前必须：
> 1. 在目标数据集上完整训练并评估 convert 真实 Q/DQ 精度；
> 2. 通过板端评估后才能作为交付结论。
```

## 4. 命令块规范

- 代码块语言固定为 `bash`。
- 统一使用环境前缀 `env PYTHONPATH="$PWD" \`；需要指定 GPU 顺序时加 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。
- 命令换行用 `\`，参数每行一个并缩进 2 个空格，保持可复制：
- 训练、评估、导出、验证分别使用项目统一入口（如 `train_qat.py`、`eval.py`、`export.py`、`test.py`），不直接调用内部实现。
- 参数值中的路径、配置名写全，不省略为“同上”。

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python export.py \
  --task detect --model yolo26n.yaml --pretrained weights/yolo26n.pt \
  --qat-weights runs/detect/yolo26n-qat-throughput/weights/best.pt \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --out yolo26_onnx/yolo26n_qat_throughput.onnx \
  --device cuda:0 --imgsz 640 640 --end2end true
```

## 5. 表格与链接

- 表格使用标准 Markdown 语法，表头与内容风格一致；数值列右对齐（`---:`），文本列左对齐。
- 配置名、模式名加反引号；同一列内描述长度尽量接近，便于阅读。
- 表格前的引语句子以冒号结尾；表格后注脚以 `注：` 开头。
- 跨文档链接使用相对路径，如 `[qat_deployment.md](./axera-npu/qat_deployment.md)`；skill 使用 `[$skill-yolo26-qat-delivery](.codex/skills/yolo26-qat-delivery/SKILL.md)`。

## 6. 完整模板

新项目可直接复制以下骨架替换内容：

````markdown
# <项目名>

本仓库基于 [<上游项目>](<链接地址>)，提供 <能力范围>。

操作建议：<端到端最短路径，含 smoke → 评估 → 导出 → 部署 → 全量训练>。

## 精度结果

### <任务 A>

<指标口径与环境说明>：

| 模型 | 模式 | 配置 | 基线指标 | 量化指标 | 误差 | 耗时 |
|---|---|---:|---:|---:|---:|---:|
| <模型> | `<模式>` | `<配置>` | 0.00 | 0.00 | 0.00 | 0.00 |

注：<术语解释、数据规模、测试环境>。

## <配置章节>

<已验证组合与适用范围>；详细流程见[<文档>](<链接地址>)。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. 准备数据集

<数据配置说明与校验方法>。

### 3. 运行 Smoke

```bash
env PYTHONPATH="$PWD" \
  python <训练入口> \
  --epochs 1 --batch 2 --imgsz 640 --workers 0 --fraction 0.01 \
  --name <smoke 名称> --exist-ok
```

### 4. 正式训练

<训练说明>。

#### <任务 A>

<任务说明>：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python <训练入口> \
  --task <任务> --model <模型 yaml> --pretrained <浮点权重> \
  --data <数据 yaml> --quant-config <配置 json> \
  --device 0 --name <实验名>
```

- **输出契约**：<部署输出张量>。
- **已支持**：<已接入能力>。
- **未接入**：<未完成项>。
- **⚠️ 仅链路验收，精度未调优**：<适用范围限制>。

### 5. 评估精度

<评估链说明：基线 → fake-quant → 真实 Q/DQ → 部署图>。

#### <任务 A>

```bash
env PYTHONPATH="$PWD" \
  python <评估入口> \
  --ckpt <checkpoint> --quant-config <配置 json> \
  --data <数据 yaml> --device cuda:0 --imgsz 640
```

### 6. 导出 <部署格式>

<导出约束说明>。

#### <任务 A>

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python <导出入口> \
  --task <任务> --model <模型 yaml> --pretrained <浮点权重> \
  --qat-weights <QAT checkpoint> --quant-config <配置 json> \
  --out <输出路径> --device cuda:0 --imgsz 640 640
```

### 7. 验收 <部署格式>

```bash
python <验收脚本> <输出模型> --quant-config <配置 json> --ort
```

#### 任务输出契约

| 任务 | 输出 | 评估入口 |
|---|---|---|
| <任务 A> | <输出张量> | `eval.py <模式>` |

### 8. 图片验证

<验证入口与保存位置说明>。

#### <任务 A>

```bash
env PYTHONPATH="$PWD" \
  python test.py \
  --model <模型或 checkpoint> --source <图片> --device cpu
```

## 模型部署

请阅读 [<部署文档>](<链接地址>)。
````

## 7. 提交前自查清单

- [ ] H1 项目名、简介、操作建议三段齐全。
- [ ] 精度结果按任务分节，表格含基线/量化/误差/耗时，并注明口径与术语。
- [ ] 快速开始步骤编号连续，任务顺序始终为检测 → 分割 → OBB → Pose → 分类 → 多 GPU。
- [ ] 每条命令可直接复制执行，环境前缀与续行缩进统一。
- [ ] 输出契约有表格汇总；未验证范围使用 `⚠️` 与 `- **未接入**` 标注。
- [ ] 配置细节、节点发现、部署步骤链接到专门文档，不在 README 内重复展开。
