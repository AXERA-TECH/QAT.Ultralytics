# QAT.Ultralytics

本仓库基于 [Ultralytics-8.4](https://github.com/ultralytics/ultralytics/tree/v8.4.21)，提供
`YOLO26` 和 `YOLO11` 系列 PT2E QAT 的训练、QuantONNX 导出和 AXERA 部署兼容处理。

操作建议：初期请选用小批量数据集，仅训练 1 个 epoch；随后依次执行 eval.py 与 export.py 完成验证及模型导出，最后进行板端部署。务必先完整走通上述端到端流程，确认无误后再启动全量数据训练。

## 精度结果

### 检测

以下为当前 AXERA NPU 实测结果：

| 模型 | `end2end` | 配置 | FP32 mAP50-95 | FP32 mAP50 | QAT mAP50-95 | QAT mAP50 | Err mAP:50~95 | Err mAP:50 | Speed(ms) |
|---|---|---|---|---|---|---|---|---|---|
| YOLO26n | `true` | ptq(w8a8_siluInU16) | 40.24 | 55.79 | 37.83 | 53.54 | -2.41 | -2.25 | 3.613 |
| YOLO26n | `true` | `config_siluInU16_attnS8_clsU16.json` | 40.24 | 55.79 | 39.61 | 55.63 | -0.63 | -0.16 | 3.761 |
| YOLO26n | `true` | `config_siluInU8_attnS8_clsU16.json` | 40.24 | 55.79 | 39.39 | 55.37 | -0.85 | -0.42 | 3.656 |
| YOLO26n | `false` | ptq(w8a8_siluInU16) | 40.87 | 56.87 | 39.52 | 55.78 | -1.35  | -1.09 | 3.616 |
| YOLO26n | `false` | `config_siluInU8_attnS8_clsU16_one2many.json` | 40.87 | 56.87 | 39.97 | 56.57 | -0.9 | -0.3 | 3.647 |
| YOLO11n | `None` | ptq(w8a8_siluInU16) | 39.4 | 55.3 | 38.8 | 54.55 | -0.6 | -0.75 | 3.934 |
| YOLO11n | `None` | `config_yolo11n_siluInU8_attnS8.json` | 39.4 | 55.3 | 38.45 | 54.46 | -0.95 | -0.84 | 3.814 |
| YOLO11n | `None` | `config_yolo11n_siluInU16_attnS8.json` | 39.4 | 55.3 | 38.84 | 54.86 | -0.56 | -0.44 | 3.851 |

性能：AX650N NPU1 模式，采用 `ax_run_model -w 10 -r 100 -m <model>.axmodel` 测得。

注：`end2end=true` 为 `one2one`模型，后处理无`nms`；`end2end=false`为`one2many`模型，后处理需`nms`。`clsU16`指的是P4、P5分类头`data_type=U16`，权重还是`s8`。

### 分割

以下为 COCO-Seg 验证集上的 QAT 训练内最优结果。`seg_best` 已完成 QuantONNX 导出与 ORT 推理。

| 模型 | `end2end` | 配置 | FP32 Box mAP50-95 | QAT Box mAP50-95 | Box 误差 | FP32 Mask mAP50-95 | QAT Mask mAP50-95 | Mask 误差 | epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n-seg | `true` | `config_yolo26nSeg_siluInU16_attnS8.json` | 39.75 | 38.86 | -0.90 | 33.86 | 33.20 | -0.66 | 44（Box 峰值） |
| YOLO26n-seg | `true` | `config_yolo26nSeg_siluInU16_attnS8.json` | 39.75 | 38.83 | -0.92 | 33.86 | 33.32 | -0.54 | 15（Mask 峰值） |

注：分割模型未上板测试，该指标为训练指标。

## QAT 量化配置

已验证的配置组合、SiLU/Attention/分类头量化边界，以及自定义网络重新发现节点的流程统一见
[AXERA QAT Quantizer 配置说明](./axera-npu/quantizer_configuration.md#31-qat-配置与已验证组合)。JSON 文件保留在
`config-qat/`；`train_qat.py` 的 `--profile accuracy|throughput` 仅对应 YOLO26n 的两个交付 profile，YOLO11、
分割和自定义网络必须显式传入 `--quant-config <json>`。

详细训练、导出和 ONNX 验收流程见[$skill-yolo26-qat-delivery](.codex/skills/yolo26-qat-delivery/SKILL.md)。
网络结构或导出环境变化后，使用[$skill-yolo-qat-config-discovery](.codex/skills/yolo-qat-config-discovery/SKILL.md)
重新发现 regional 节点。
新增 OBB、Pose、分类或自定义 head 等网络任务时，先使用
[$skill-yolo-qat-task-onboarding](.codex/skills/yolo-qat-task-onboarding/SKILL.md) 确认 trainer、loss、
validator、QAT 输出与导出契约，再开始配置和精度实验；不要将检测模型的输出 wrapper 或 regional 节点直接复用。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. 准备数据集

确认数据集配置可用。默认脚本依赖 `coco.yaml`，需要保证其中的数据集路径正确。

### 3. 运行 Smoke

```bash
env PYTHONPATH="$PWD" \
  python train_qat.py \
  --profile throughput --device 0 --epochs 1 --batch 2 --imgsz 64 \
  --workers 0 --fraction 0.01 --name qat-delivery-smoke --exist-ok
```

### 4. 正式训练

单卡入口支持检测、分割、OBB、Pose 和分类。正式训练默认使用完整数据集（`fraction=1.0`），
`--fraction 0.01` 仅用于 smoke。未提供 `--quant-config` 或 `--profile` 会直接报错。

> **⚠️ 精度验证范围说明**
>
> **检测**（YOLO26n/YOLO11n）和**分割**（YOLO26n-seg）已完成正式精度实验，量化配置经过系统优化并通过上板精度评估，可作为交付基线。
>
> **OBB、Pose 和分类**目前仅完成主机侧链路 smoke（1 epoch、小数据集），当前量化配置（全局 SiLU U8 + Attention S8）**未经正式精度调优**，不保证在实际业务数据上能满足精度要求。使用前必须：
> 1. 在目标数据集上完整训练并评估 convert 真实 Q/DQ 精度；
> 2. 若损失超出可接受范围，按 `$skill-yolo-qat-config-discovery` 重新发现局部量化边界；
> 3. 通过 AXERA 板端评估后才能作为交付结论。

#### YOLO26 检测

```bash
env PYTHONPATH="$PWD" \
  python train_qat.py \
  --profile throughput --device 0
```

#### YOLO11 检测

YOLO11n 使用专用 JSON 显式启动训练，不使用仅面向 YOLO26n 的 `--profile`：

```bash
env PYTHONPATH="$PWD" \
  python train_qat.py \
  --task detect --model yolo11n.yaml --pretrained yolo11n.pt --data coco.yaml \
  --quant-config config-qat/config_yolo11n_siluInU8_attnS8.json \
  --device 0 --name yolo11n-qat-siluInU8-attnS8
```

#### 分割

当前已验证的 YOLO26n-seg 训练参数如下：

```bash
env PYTHONPATH="$PWD" \
  python train_qat.py \
  --task segment --model yolo26n-seg.yaml --pretrained weights/yolo26n-seg.pt \
  --data coco-seg.yaml --quant-config config-qat/config_yolo26nSeg_siluInU16_attnS8.json \
  --epochs 50 --batch 64 --imgsz 640 --workers 8 --fraction 1.0 \
  --lr0 2e-5 --lrf 0.1 --no-qat-ema --end2end --save-period 1 \
  --device 0 --name yolo26n-seg-qat
```

#### OBB

YOLO26n-OBB 已完成 `dota8` 1 epoch 的训练、QAT/真实 Q-DQ 评估、checkpoint/QuantONNX 图片推理和 ONNX
导出 smoke。使用 DOTA 格式数据和 OBB 专用浮点权重，不要使用检测任务的 COCO 数据、分类头 U16 配置或导出
wrapper：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python train_qat.py \
  --task obb --model yolo26n-obb.yaml --pretrained weights/yolo26n-obb.pt \
  --data dota8.yaml --quant-config config-qat/config_yolo26nObb_siluInU8_attnS8.json \
  --device 0 --epochs 1 --batch 1 --imgsz 640 --workers 0 \
  --fraction 1.0 --name yolo26n-obb-qat-smoke --exist-ok
```

启动日志应显示 `Transferred 792/792 items from pretrained weights`；否则先检查 DOTA 15 类 head 是否正确加载。

- **Smoke 指标**：`dota8` 上的 1 epoch smoke，QAT `mAP50-95/mAP50=0.856/0.995`，浮点为 `0.891/0.995`。数据集仅含 4 张验证图，结果只用于链路验收。
- **输出契约**：OBB QuantONNX 输出为 `boxes`、`scores`、`angle`。
- **已支持**：`test.py --task obb` 图片验证、`eval.py onnx-obb` QuantONNX 数据集评估和 Pulsar2 转换配置（`axera-npu/config_yolo26nObb_siluInU8_attnS8.json`，Attention S8）。
- **⚠️ 仅链路验收，精度未调优**：当前量化配置未经正式精度优化，上板精度结论需使用目标 DOTA 完整数据集重新训练验证。
- **未接入**：AXERA 旋转框后处理、rotated NMS 和板端精度评估。

#### Pose

YOLO26n-Pose 已打通主机侧 QAT 训练、评估、导出和图片验证。使用 discovery 生成的 attention S8 配置
`config-qat/config_yolo26nPose_siluInU8_attnS8.json`（全局 SiLU U8 + 双 Attention 连续 S8）：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python train_qat.py \
  --task pose --model yolo26n-pose.yaml --pretrained weights/yolo26n-pose.pt \
  --data coco8-pose.yaml --quant-config config-qat/config_yolo26nPose_siluInU8_attnS8.json \
  --device 3 --epochs 1 --batch 1 --imgsz 640 --workers 0 \
  --fraction 1.0 --name yolo26n-pose-qat-smoke --exist-ok
```

- **输出契约**：Pose QuantONNX 输出 `boxes`、`scores`、`keypoints`，训练专用的 `kpts_sigma` 不导出。
- **Smoke 指标**：`coco8-pose` 1 epoch smoke 的 Float、fake-quant、真实 Q/DQ、QuantONNX Box/Pose mAP50-95 分别为 `0.7006/0.4773`、`0.6753/0.4040`、`0.7204/0.4380`、`0.7085/0.4490`，仅用于链路验收。
- **已支持**：Pulsar2 转换配置（`axera-npu/config_yolo26nPose_siluInU8_attnS8.json`，Attention S8）。
- **⚠️ 仅链路验收，精度未调优**：当前量化配置未经正式精度优化，上板精度结论需使用目标完整数据集重新训练验证。
- **未接入**：AXERA 关键点后处理和板端精度评估。

#### 分类

YOLO26n-Cls 已打通主机侧 QAT 训练、评估、导出和图片验证。分类骨干含一个 `C2PSA`，使用 discovery 生成的
attention S8 配置 `config-qat/config_yolo26nCls_siluInU8_attnS8.json`（全局 SiLU U8 + 单 Attention 连续 S8）。
smoke 数据集用最小 `imagenet10`（分类微调会把 head 重塑到数据集 `nc`，指标仅作链路验收）：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python train_qat.py \
  --task classify --model yolo26n-cls.yaml --pretrained weights/yolo26n-cls.pt \
  --data imagenet10 --quant-config config-qat/config_yolo26nCls_siluInU8_attnS8.json \
  --device 0 --epochs 1 --batch 32 --imgsz 224 --workers 4 \
  --fraction 1.0 --name yolo26n-cls-qat-smoke --exist-ok
```

- **输出契约**：分类 QuantONNX 输出单个 `logits` 张量。`ClassifyWrapper` 不追加 softmax（softmax 不在 QAT 图内，追加会留下未量化的尾算子），部署时 host 端做 softmax/argmax；`argmax(logits)==argmax(softmax)`。
- **Smoke 指标**：`imagenet10` 上真实 Q/DQ `top1/top5=0.0833/0.6667`。数据仅 12 张验证图，结果只用于链路验收。
- **部署图结构**：ONNX checker/ORT 通过，`BatchNormalization=0`、`_requant Identity=0`、无游离 S16，Attention 四点连续 S8，尾部收在 `logits` 的 DequantizeLinear。
- **已支持**：Pulsar2 转换配置（`axera-npu/config_yolo26nCls_siluInU8_attnS8.json`，Attention S8）。
- **⚠️ 仅链路验收，精度未调优**：当前量化配置未经正式精度优化，上板精度结论需使用目标 ImageNet 完整数据集重新训练验证。
- **未接入**：AXERA 分类 Top-K 后处理和板端精度评估。
#### 多 GPU 检测

实验性多 GPU QAT 使用独立入口。`--batch` 是全局 batch，必须能被 GPU 数量整除；先用小数据 smoke
确认显存、DDP 构图和 checkpoint 导出正常，再进行正式训练：

```bash
env PYTHONPATH="$PWD" \
  python train_gpus.py \
  --profile throughput --devices 0,1 \
  --epochs 1 --batch 4 --imgsz 64 --workers 0 --fraction 0.01 \
  --name qat-ddp-smoke --exist-ok
```

两卡正式检测训练示例。`--batch 64` 是全局 batch，每个 rank 使用 32；使用自定义模型时替换
`--model`、`--pretrained`、`--data` 与 `--quant-config`，并先重新发现 regional 节点：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python train_gpus.py \
  --task detect --model yolo26n.yaml --pretrained weights/yolo26n.pt --data coco.yaml \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --devices 0,1 --epochs 50 --batch 64 --imgsz 640 --workers 4 --fraction 1.0 \
  --lr0 2e-5 --lrf 0.1 --no-qat-ema --end2end --save-period 1 \
  --name yolo26n-qat-throughput-ddp
```

`--devices` 接受至少两张、不重复的 GPU；全局 `--batch` 必须能被 GPU 数量整除，`--workers` 按每个 rank 计算。
`--quant-config` 可替代 `--profile`，且优先级更高。

当前仅验证双卡检测 QAT。三卡以上、多机和分割 DDP 尚未验证；交付前须与单卡对齐 `convert_pt2e` 精度和 ONNX 结构。

Quantizer 配置能力及自定义网络注意事项见[quantizer_configuration.md](./axera-npu/quantizer_configuration.md)。

### 5. 评估精度

评估链：`float`（浮点基线）→ `qat`（fake-quant 对照）→ `convert`（真实 Q/DQ，**交付判定依据**）→ QuantONNX。
`qat` 使用 checkpoint 中的冻结 qparams 重建 prepared 图，指标应能精确复现训练内 validation 结果。
运行 `python eval.py <mode> --help` 查看完整参数。

#### 检测

```bash
# 浮点基线
env PYTHONPATH="$PWD" \
  python eval.py float \
  --task detect --weights weights/yolo26n.pt \
  --data coco.yaml --device cuda:0 --imgsz 640

# fake-quant 对照（指标应与训练内 validation 一致）
env PYTHONPATH="$PWD" \
  python eval.py qat \
  --ckpt runs/detect/yolo26n-qat-throughput/weights/best.pt \
  --model yolo26n.yaml --pretrained weights/yolo26n.pt \
  --data coco.yaml --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --device cuda:0 --imgsz 640

# 真实 Q/DQ（交付判定）
env PYTHONPATH="$PWD" \
  python eval.py convert \
  --ckpt runs/detect/yolo26n-qat-throughput/weights/best.pt \
  --model yolo26n.yaml --pretrained weights/yolo26n.pt \
  --data coco.yaml --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --device cuda:0 --imgsz 640

# QuantONNX（ORT，须传 --model/--pretrained 以获取 head 结构）
env PYTHONPATH="$PWD" \
  python eval.py onnx \
  --onnx yolo26_onnx/yolo26n_qat_throughput_slim.onnx \
  --model yolo26n.yaml --pretrained weights/yolo26n.pt \
  --data coco.yaml --device cpu --imgsz 640 --batch 16
```

YOLO11n 将 `--model` 换为 `yolo11n.yaml`、`--pretrained` 换为 `weights/yolo11n.pt`，其余参数相同；
`eval.py onnx` 会从 head 自动读取 `reg_max=16` 进行 DFL 解码，无需单独后端。
`onnx`、`onnx-obb` 和 `onnx-pose` 默认从 QuantONNX 的固定输入 shape 获取尺寸；矩形模型可显式传入
`--imgsz 352 640`（顺序为 `height width`），但必须与 ONNX 输入完全一致。

#### 分割

分割 QAT checkpoint 使用独立的 `segment` 模式，评估会从 checkpoint 恢复训练时的 `overlap_mask` 和
`mask_ratio`，QAT state 必须与当前 prepared graph 严格匹配：

```bash
env PYTHONPATH="$PWD" \
  python eval.py segment \
  --ckpt runs/segment/yolo26n-seg-qat/weights/best.pt \
  --model yolo26n-seg.yaml --pretrained weights/yolo26n-seg.pt \
  --quant-config config-qat/config_yolo26nSeg_siluInU16_attnS8.json \
  --data coco-seg.yaml --device 0 --batch 16
```

#### OBB

OBB checkpoint 使用 `qat`/`convert` 评估旋转框 mAP；QuantONNX 三输出（`boxes`、`scores`、`angle`）
使用 `onnx-obb`，不支持 rect validation：

```bash
# fake-quant 对照
env PYTHONPATH="$PWD" \
  python eval.py qat \
  --task obb --ckpt runs/obb/yolo26n-obb-qat-smoke/weights/best.pt \
  --model yolo26n-obb.yaml --pretrained weights/yolo26n-obb.pt \
  --data dota8.yaml --quant-config config-qat/config_yolo26nObb_siluInU8_attnS8.json \
  --device cuda:0 --imgsz 640 --pycoco False

# 真实 Q/DQ（交付判定）
env PYTHONPATH="$PWD" \
  python eval.py convert \
  --task obb --ckpt runs/obb/yolo26n-obb-qat-smoke/weights/best.pt \
  --model yolo26n-obb.yaml --pretrained weights/yolo26n-obb.pt \
  --data dota8.yaml --quant-config config-qat/config_yolo26nObb_siluInU8_attnS8.json \
  --device cuda:0 --imgsz 640 --pycoco False

# QuantONNX
env PYTHONPATH="$PWD" \
  python eval.py onnx-obb \
  --onnx yolo26_onnx/yolo26n_obb_qat_slim.onnx \
  --model yolo26n-obb.yaml --pretrained weights/yolo26n-obb.pt \
  --data dota8.yaml --device cpu --imgsz 640 --batch 16 --workers 4
```

#### Pose

Pose checkpoint 使用 `qat`/`convert` 评估；QuantONNX 三输出（`boxes`、`scores`、`keypoints`）
使用 `onnx-pose` 同时评估 Box/Pose mAP：

```bash
# fake-quant 对照
env PYTHONPATH="$PWD" \
  python eval.py qat \
  --task pose --ckpt runs/pose/yolo26n-pose-qat-smoke/weights/best.pt \
  --model yolo26n-pose.yaml --pretrained weights/yolo26n-pose.pt \
  --data coco8-pose.yaml --quant-config config-qat/config_yolo26nPose_siluInU8_attnS8.json \
  --device cuda:0 --imgsz 640 --pycoco False

# 真实 Q/DQ（交付判定）
env PYTHONPATH="$PWD" \
  python eval.py convert \
  --task pose --ckpt runs/pose/yolo26n-pose-qat-smoke/weights/best.pt \
  --model yolo26n-pose.yaml --pretrained weights/yolo26n-pose.pt \
  --data coco8-pose.yaml --quant-config config-qat/config_yolo26nPose_siluInU8_attnS8.json \
  --device cuda:0 --imgsz 640 --pycoco False

# QuantONNX
env PYTHONPATH="$PWD" \
  python eval.py onnx-pose \
  --onnx yolo26_onnx/yolo26n_pose_qat_smoke_slim.onnx \
  --model yolo26n-pose.yaml --pretrained weights/yolo26n-pose.pt \
  --data coco8-pose.yaml --device cpu --imgsz 640 --batch 1 --workers 0
```

#### 分类

分类指标为 Top-1/Top-5；QuantONNX 输出单个 `logits`，host 端做 softmax/argmax：

```bash
# fake-quant 对照
env PYTHONPATH="$PWD" \
  python eval.py qat \
  --task classify --ckpt runs/classify/yolo26n-cls-qat-smoke/weights/best.pt \
  --model yolo26n-cls.yaml --pretrained weights/yolo26n-cls.pt \
  --data imagenet10 --quant-config config-qat/config_yolo26nCls_siluInU8_attnS8.json \
  --device cuda:0 --imgsz 224 --batch 32 --workers 4

# 真实 Q/DQ（交付判定）
env PYTHONPATH="$PWD" \
  python eval.py convert \
  --task classify --ckpt runs/classify/yolo26n-cls-qat-smoke/weights/best.pt \
  --model yolo26n-cls.yaml --pretrained weights/yolo26n-cls.pt \
  --data imagenet10 --quant-config config-qat/config_yolo26nCls_siluInU8_attnS8.json \
  --device cuda:0 --imgsz 224 --batch 32 --workers 4
```

### 6. 导出 QuantONNX

QAT checkpoint 必须使用 `export.py`，显式指定任务、浮点权重、QAT checkpoint 和量化配置。导出失败
说明 checkpoint、配置或量化图存在问题，禁止修改模型加载逻辑强行导出。`export.py` 会在 `_slim.onnx`
中自动完成 BN 折叠、requant 合并和 Split/Reshape 对齐后处理。`--end2end` 控制选 one2one（NMS-free）
还是 one2many 分支；Segment 和 Classify 固定使用 one2one，忽略该参数。

#### 检测

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python export.py \
  --task detect --model yolo26n.yaml --pretrained weights/yolo26n.pt \
  --qat-weights runs/detect/yolo26n-qat-throughput/weights/best.pt \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --out yolo26_onnx/yolo26n_qat_throughput.onnx \
  --device cuda:0 --imgsz 640 640 --end2end true
```

#### 分割

输出 9 个张量：三尺度 `boxes_pN/scores_pN`、`mask_coefficient`、`proto_masks`、`proto_semseg`：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python export.py \
  --task segment --model yolo26n-seg.yaml --pretrained weights/yolo26n-seg.pt \
  --qat-weights runs/segment/yolo26n-seg-qat/weights/best.pt \
  --quant-config config-qat/config_yolo26nSeg_siluInU16_attnS8.json \
  --out yolo26_onnx/yolo26n_seg_qat.onnx \
  --device cuda:0 --imgsz 640 640
```

#### OBB

输出 3 个张量：`boxes`、`scores`、`angle`：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python export.py \
  --task obb --model yolo26n-obb.yaml --pretrained weights/yolo26n-obb.pt \
  --qat-weights runs/obb/yolo26n-obb-qat-smoke/weights/best.pt \
  --quant-config config-qat/config_yolo26nObb_siluInU8_attnS8.json \
  --out yolo26_onnx/yolo26n_obb_qat.onnx \
  --device cuda:0 --imgsz 640 640 --end2end true
```

#### Pose

输出 3 个张量：`boxes`、`scores`、`keypoints`：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python export.py \
  --task pose --model yolo26n-pose.yaml --pretrained weights/yolo26n-pose.pt \
  --qat-weights runs/pose/yolo26n-pose-qat-smoke/weights/best.pt \
  --quant-config config-qat/config_yolo26nPose_siluInU8_attnS8.json \
  --out yolo26_onnx/yolo26n_pose_qat.onnx \
  --device cuda:0 --imgsz 640 640 --end2end true
```

#### 分类

不使用 `--end2end`，输入 224，输出单个 `logits`（host 端做 softmax/argmax）：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python export.py \
  --task classify --model yolo26n-cls.yaml --pretrained weights/yolo26n-cls.pt \
  --qat-weights runs/classify/yolo26n-cls-qat-smoke/weights/best.pt \
  --quant-config config-qat/config_yolo26nCls_siluInU8_attnS8.json \
  --out yolo26_onnx/yolo26n_cls_qat_smoke.onnx \
  --device cuda:0 --imgsz 224 224
```

### 7. 验收 Slim ONNX

#### 检测结构验收

```bash
python \
  .codex/skills/yolo-qat-config-discovery/scripts/validate_qat_structure.py \
  yolo26_onnx/yolo26n_qat_throughput_slim.onnx \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --ort --expect-aligned-split-reshape 2
```

#### 任务输出契约

| 任务 | QuantONNX 输出 | 评估入口 |
|---|---|---|
| 检测 | 三尺度 `boxes/scores`，共 6 个输出 | `eval.py onnx` |
| 分割 | 三尺度 `boxes/scores`、mask coefficient、proto | `eval.py segment`（checkpoint） |
| OBB | `boxes`、`scores`、`angle` | `eval.py onnx-obb` |
| Pose | `boxes`、`scores`、`keypoints` | `eval.py onnx-pose` |
| 分类 | 单个 `logits`（host 端做 softmax/argmax） | `eval.py convert`（checkpoint） |

所有最终模型均需通过 ONNX checker 和目标运行时加载。检测交付模型还必须满足 BN、requant、Attention S8 和
Split/Reshape 量化参数约束；其他任务按各自训练配置和输出契约验收，不得直接套用检测节点编号。

### 8. 图片验证

`test.py` 支持单张图片、目录或 glob，绘制结果默认保存到 `runs/predict/qat-test/`。使用
`--save-txt --save-conf` 可同时保存标签。`--imgsz 640` 表示方形输入，矩形 QAT checkpoint 可使用
`--imgsz 352 640`（顺序为 `height width`）；QuantONNX 会直接采用模型中记录的固定输入尺寸。

#### 检测

```bash
# QAT checkpoint：自动读取 checkpoint 中记录的 quant config
env PYTHONPATH="$PWD" \
  python test.py \
  --model runs/detect/yolo26n-qat-throughput/weights/best.pt \
  --source bus.jpg --device cpu

# 六输出 QuantONNX
env PYTHONPATH="$PWD" \
  python test.py \
  --model yolo26_onnx/yolo26n_qat_throughput_slim.onnx \
  --source bus.jpg --device cpu
```

#### 分割

分割 checkpoint 和 QuantONNX 使用相同入口，增加 `--task segment`。输出图同时绘制检测框和实例 mask：

```bash
env PYTHONPATH="$PWD" \
  python test.py --task segment \
  --model runs/segment/yolo26n-seg-qat/weights/best.pt \
  --model-yaml yolo26n-seg.yaml --pretrained weights/yolo26n-seg.pt \
  --source bus.jpg --device 0

env PYTHONPATH="$PWD" \
  python test.py --task segment \
  --model yolo26_onnx/yolo26n_seg_qat_slim.onnx \
  --model-yaml yolo26n-seg.yaml --pretrained weights/yolo26n-seg.pt \
  --source bus.jpg --device cpu
```

#### OBB

OBB checkpoint 和三输出 QuantONNX 使用 `--task obb`。输出会绘制旋转框；OBB QuantONNX 必须保留
`boxes`、`scores`、`angle` 三个拼接输出：

```bash
env PYTHONPATH="$PWD" \
  python test.py --task obb \
  --model runs/obb/yolo26n-obb-qat-attnS8-smoke/weights/best.pt \
  --model-yaml yolo26n-obb.yaml --pretrained weights/yolo26n-obb.pt \
  --quant-config config-qat/config_yolo26nObb_siluInU8_attnS8.json \
  --source bus.jpg --device cpu

env PYTHONPATH="$PWD" \
  python test.py --task obb \
  --model yolo26_onnx/yolo26n_obb_qat_attnS8_smoke_slim.onnx \
  --model-yaml yolo26n-obb.yaml --pretrained weights/yolo26n-obb.pt \
  --source bus.jpg --device cpu
```

#### Pose

Pose checkpoint 和三输出 QuantONNX 使用 `--task pose`。输出会绘制人体关键点；Pose QuantONNX 必须保留
`boxes`、`scores`、`keypoints` 三个拼接输出：

```bash
env PYTHONPATH="$PWD" \
  python test.py --task pose \
  --model runs/pose/yolo26n-pose-qat-attnS8-smoke/weights/best.pt \
  --model-yaml yolo26n-pose.yaml --pretrained weights/yolo26n-pose.pt \
  --quant-config config-qat/config_yolo26nPose_siluInU8_attnS8.json \
  --source bus.jpg --device cpu

env PYTHONPATH="$PWD" \
  python test.py --task pose \
  --model yolo26_onnx/yolo26n_pose_qat_slim.onnx \
  --model-yaml yolo26n-pose.yaml --pretrained weights/yolo26n-pose.pt \
  --source bus.jpg --device cpu
```

#### 分类

分类 checkpoint 和单输出 QuantONNX 使用 `--task classify`。输出绘制 Top-5 类名与概率（ONNX 输出 `logits`，
test.py 在 host 端做 softmax 后显示）；`--data` 提供数据集名以显示正确类名（checkpoint 会自动从 `train_args`
读取）：

```bash
env PYTHONPATH="$PWD" \
  python test.py --task classify \
  --model runs/classify/yolo26n-cls-qat-smoke/weights/best.pt \
  --model-yaml yolo26n-cls.yaml --pretrained weights/yolo26n-cls.pt \
  --quant-config config-qat/config_yolo26nCls_siluInU8_attnS8.json \
  --source bus.jpg --device cpu

env PYTHONPATH="$PWD" \
  python test.py --task classify \
  --model yolo26_onnx/yolo26n_cls_qat_smoke_slim.onnx \
  --data imagenet10 --source bus.jpg --device cpu
```

## 模型部署
请阅读 [qat_deployment.md](./axera-npu/qat_deployment.md)。
