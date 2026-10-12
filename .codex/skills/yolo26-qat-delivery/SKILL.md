---
name: yolo26-qat-delivery
description: 在本仓库训练、导出和验收最终 YOLO26n PT2E QAT 交付模型。用于客户需要选择 accuracy/throughput 配置、启动 QAT、通过 export.py 导出 ONNX，或检查 Attention S8、Split/Reshape、BN 和 requant 结构时。
---

# YOLO26 QAT Delivery

`accuracy` 和 `throughput` 两个交付 profile 仅适用于 YOLO26n 检测模型、`end2end=True`。不要恢复
历史实验配置，也不要通过放宽 `state_dict` 加载或修改模型代码来绕过导出错误。

模型结构、PyTorch 版本或导出环境变化后，先使用 `$yolo-qat-config-discovery` 重新发现分类塔 U16
和 Attention S8 节点；不得直接复用旧 FX 节点编号。

## 环境

- 在仓库根目录执行命令。
- 使用项目兼容的 PyTorch 2.6 QAT 环境，命令执行前先激活，并通过 `python -c "import torch"` 确认可用。
- 设置 `PYTHONPATH="$PWD"`。
- 训练和导出前确认目标 GPU 空闲；不要中断其他运行中的训练。

## 选择 Profile

- 精度余量优先：选择 `accuracy`，配置为 `config-qat/config_siluInU16_attnS8_clsU16.json`。
- 吞吐和激活带宽优先：选择 `throughput`，配置为 `config-qat/config_siluInU8_attnS8_clsU16.json`。
- 读取 [references/profiles.md](references/profiles.md) 获取两者量化边界和验收标准。

## 训练

先运行小规模 smoke，再启动正式训练：

```bash
env PYTHONPATH="$PWD" \
  python train_qat.py \
  --profile throughput --device 0 --epochs 1 --batch 2 --imgsz 640 \
  --workers 0 --fraction 0.01 --name qat-delivery-smoke --exist-ok
```

正式训练去掉 smoke 参数，保留默认 50 epoch、640 输入和 noEMA。不得加载其他 profile 的 checkpoint
续训；profile 或图结构变化后必须从浮点预训练权重重新开始。

## 训练变体

### One-to-many

- YOLO26 one-to-many 使用独立配置：先按 `$yolo-qat-config-discovery` 以 `--branch cv3` 重新发现节点，
  再以 `train_qat.py --no-end2end` 训练，并以 `export.py --end2end false` 导出。不要复用 one-to-one 的
  clsU16 节点。
- one-to-many 的分类 score 是 Sigmoid 后的 U8；分类 logit 的 U16 在 Sigmoid 前生效。结构检查使用
  `--skip-output-check`，输出契约另行验收。

### 分割

- YOLO26n-seg 不使用本节 profile，使用
  `config-qat/config_yolo26nSeg_siluInU16_attnS8.json`（全局 SiLU input U16、Attention S8、无 clsU16）。
  导出命令将 `--task detect` 改为 `--task segment`，并使用分割训练权重和配置。

## 导出

始终使用 `export.py`，显式指定 checkpoint、配置和输出路径：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python export.py \
  --task detect \
  --model yolo26n.yaml \
  --pretrained yolo26n.pt \
  --qat-weights runs/detect/yolo26n-qat-throughput/weights/best.pt \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --out yolo26_onnx/yolo26n_qat_throughput.onnx \
  --device cuda:0 \
  --imgsz 640 640 \
  --end2end true
```

保持严格 checkpoint 加载。导不出时先检查 profile、batch 构图和 checkpoint 是否匹配，不要修改
`ultralytics/utils` 来迁就异常权重。默认不导出 `.pth`；只有客户明确需要 QAT state 时才使用
`--export-pth`。

## 评估

统一使用仓库根目录的 `eval.py`，不要直接调用 `scripts/eval_backends` 中的内部实现。交付精度以
`convert_pt2e` 后的真实 Q/DQ 结果为准：

```bash
env PYTHONPATH="$PWD" \
  python eval.py convert \
  --ckpt runs/detect/yolo26n-qat-throughput/weights/best.pt \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --device cuda:0
```

`qat` 用于核对 prepared fake-quant，`onnx` 用于核对六输出 one2one QuantONNX，`float` 用于浮点
基线。`segment`、`ptq` 和历史兼容的 `onnx-one2many` 也通过同一入口分发。

单图或图片目录可使用 `test.py` 检查检测框。输入 `.pt` 时脚本按 checkpoint 内的 quant config 重建
QAT graph；输入 `.onnx` 时按六输出 one2one QuantONNX 解码。不要用普通 `YOLO(best.pt)` 代替：

```bash
env PYTHONPATH="$PWD" \
  python test.py \
  --model yolo26_onnx/yolo26n_qat_throughput_slim.onnx \
  --source bus.jpg --device cpu
```

## 验收 ONNX

对 `export.py` 生成的 `_slim.onnx` 执行：

```bash
python \
  .codex/skills/yolo-qat-config-discovery/scripts/validate_qat_structure.py \
  yolo26_onnx/yolo26n_qat_throughput_slim.onnx \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --ort --expect-aligned-split-reshape 2
```

标准检测 one-to-one 交付图必须满足：

- ONNX checker 通过，ORT 可加载。
- `BatchNormalization=0`。
- 名称包含 `_requant` 的 Identity 为 0。
- 配置中的每个 Attention 均满足第一 MatMul 输出 S8、Softmax 输入输出 S8、第二 MatMul 输出 U8。
- 分类 score 输出量化位宽与配置中的 clsU16 数量一致。
- Split 输入量化范围覆盖的 Reshape 分支使用相同 scale/zero-point。

分割模型当前已知可能存在 1 个 mask-head requant，不能被此处的零 requant 规则自动忽略；应定位共享 feature
的 qspec 并通过配置、重训修复。其他任一项失败都视为交付模型异常。先核对配置和导出日志，不要手工改最终 ONNX 掩盖训练图问题；
`export.py` 内置的 Split/Reshape 对齐属于 AXERA 后端兼容处理，可以保留。

## 回归

修改量化配置、quantizer 或导出后处理后运行：

```bash
env PYTHONPATH="$PWD" \
  python -m pytest \
  tests/test_qat_observer_config.py \
  tests/test_export_qat_prepare.py \
  tests/test_qat_export_postprocess.py
```

同时运行 `git diff --check`，并重新执行一次真实 `export.py` 和 ONNX 验收。
