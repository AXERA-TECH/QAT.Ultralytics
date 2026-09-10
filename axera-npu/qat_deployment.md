# YOLO QAT 部署转换

本文适用于最终 YOLO26/YOLO11 QAT 导出的 `*_slim.onnx`。YOLO11 和自定义模型必须显式使用各自训练 JSON。

Quantizer 新增配置能力、Attention S8 量化边界及自定义 YOLO26/YOLO11 网络的兼容说明见
[quantizer_configuration.md](./quantizer_configuration.md)。

## 适用范围

| 任务 | QuantONNX/ORT | Pulsar2 配置 | AXModel 后处理 |
|---|---|---|---|
| 检测 | 已支持 | 已支持 | `run_yolo_detect.py` |
| 分割 | 已支持 | 已支持 | `run_yolo_seg.py` |
| OBB | 已支持 | 基础配置（Attention S8） | 未接入 rotated NMS |
| Pose | 已支持 | 基础配置（Attention S8） | 未接入关键点后处理 |
| 分类 | 已支持 | 基础配置（Attention S8） | 未接入 Top-K 后处理 |

> **任务输出契约**
>
> - OBB 输出为 `boxes`、`scores`、`angle`，Pose 输出为 `boxes`、`scores`、`keypoints`，分类输出单个 `logits`（host 端做 softmax/argmax），均不是检测模型的六个逐尺度输出。
> - 不得复用检测 wrapper、`run_yolo_detect.py` 或检测转换 JSON。
> - 分类及其他自定义任务接入前先执行 [$skill-yolo-qat-task-onboarding](../.codex/skills/yolo-qat-task-onboarding/SKILL.md)。

> **⚠️ 精度验证范围**
>
> **检测**已通过正式精度实验和上板评估，**分割**已通过精度实验，当前配置可作为交付基线。
> **OBB、Pose 和分类**仅完成主机侧链路 smoke，当前量化配置未经正式精度调优，**不保证满足业务精度要求**。
> 使用前须在目标完整数据集上重新训练、评估 convert 真实 Q/DQ 精度并通过 AXERA 板端验证后，才能作为交付结论。

## 1. 导出并验收 QuantONNX

使用 `export.py` 显式指定 checkpoint 和对应配置。导出完成后先执行：

```bash
python \
  .codex/skills/yolo26-qat-delivery/scripts/validate_onnx.py \
  path/to/model_slim.onnx --ort --expect-attention-s8
```

必须满足 ONNX checker/ORT 通过、BN=0、无额外 requant，且相关 Split/Reshape 量化参数已对齐。

## 2. Pulsar2 配置

### 配置对应关系

AXERA 转换配置必须与训练量化配置同名、不同目录。例如训练使用
`config-qat/config_siluInU8_attnS8_clsU16.json` 时，使用
`axera-npu/config_siluInU8_attnS8_clsU16.json`。该配置的 `input` 必须指向本次导出的精确 `_slim.onnx`，
不能保留历史实验路径。

当前已核对的对应关系：

| 模型 | 训练量化配置 | AXERA 转换配置 | Attention 数量 |
|---|---|---|---:|
| YOLO26n one-to-one、全局 SiLU U8 + Attention S8 + clsU16 | `config-qat/config_siluInU8_attnS8_clsU16.json` | `axera-npu/config_siluInU8_attnS8_clsU16.json` | 2 |
| YOLO26n one-to-many、全局 SiLU U8 + Attention S8 + clsU16 | `config-qat/config_siluInU8_attnS8_clsU16_one2many.json` | `axera-npu/config_siluInU8_attnS8_clsU16_one2many.json` | 2 |
| YOLO11n、全局 SiLU U8 + Attention S8 | `config-qat/config_yolo11n_siluInU8_attnS8.json` | `axera-npu/config_yolo11n_qat.json` | 1 |
| YOLO11n、SiLU input U16 + Attention S8 | `config-qat/config_yolo11n_siluInU16_attnS8.json` | `axera-npu/config_yolo11n_siluInU16_attnS8.json` | 1 |
| YOLO26n-seg、SiLU input U16 + Attention S8 | `config-qat/config_yolo26nSeg_siluInU16_attnS8.json` | `axera-npu/config_yolo26nSeg_siluInU16_attnS8.json` | 2 |
| YOLO26n-cls、全局 SiLU U8 + Attention S8 | `config-qat/config_yolo26nCls_siluInU8_attnS8.json` | `axera-npu/config_yolo26nCls_siluInU8_attnS8.json` | 1 |
| YOLO26n-Pose smoke、全局 SiLU U8 + Attention S8 | `config-qat/config_yolo26nPose_siluInU8_attnS8.json` | `axera-npu/config_yolo26nPose_siluInU8_attnS8.json` | 2 |
| YOLO26n-OBB smoke、全局 SiLU U8 + Attention S8 | `config-qat/config_yolo26nObb_siluInU8_attnS8.json` | `axera-npu/config_yolo26nObb_siluInU8_attnS8.json` | 2 |

- 使用 `$axera-quantonnx-config` 从该 ONNX 自动生成或复核配置，再只修改输出目录和工具链要求的输入/输出字段。
- QuantONNX 已携带 Q/DQ 参数，不需要校准集；配置中的 `/path/to/dataset` 仅为 Pulsar2 必填字段的占位值。

### Attention S8 配置生成

```bash
env PYTHONPATH="$PWD" \
  python .codex/skills/axera-quantonnx-config/scripts/generate_axera_config.py \
  --onnx yolo26_onnx/yolo26n_qat_throughput_slim.onnx \
  --output axera-npu/config_siluInU8_attnS8_clsU16.json \
  --output-dir ./output_yolo26n_siluInU8_attnS8_clsU16
```

生成器会验证 Attention S8 边界并按当前 ONNX 写入 layer override；不得复制其他模型或历史导出图的节点名。

### 无 Regional Attention S8 的任务

若模型训练配置不含 Attention S8 regional 约束，例如当前 OBB smoke 使用的 `config-qat/config.json`，则生成器会因
Attention 节点仍为 U8 而拒绝写入 S8 override。此时 Pulsar2 配置应只保留 QuantONNX 输入、全局 SiLU U8 和基础编译
字段，不要手工套用检测模型的 Attention S8 layer override。

## 3. 转换

### 检测模型

在 Pulsar2 环境中执行：

```bash
pulsar2 build \
  --input yolo26_onnx/yolo26n_qat_throughput_slim.onnx \
  --config axera-npu/config_siluInU8_attnS8_clsU16.json \
  --output_dir output/yolo26n_siluInU8_attnS8_clsU16
```

### OBB Smoke

```bash
pulsar2 build \
  --input yolo26_onnx/yolo26n_obb_qat_attnS8_smoke_slim.onnx \
  --config axera-npu/config_yolo26nObb_siluInU8_attnS8.json \
  --output_dir output_yolo26n_obb_qat_attnS8_smoke
```

### 分割

```bash
pulsar2 build \
  --input yolo26_onnx/seg_best_slim.onnx \
  --config axera-npu/config_yolo26nSeg_siluInU16_attnS8.json \
  --output_dir output/yolo26nSeg_siluInU16_attnS8
```

### 分类

```bash
pulsar2 build \
  --input yolo26_onnx/yolo26n_cls_qat_smoke_slim.onnx \
  --config axera-npu/config_yolo26nCls_siluInU8_attnS8.json \
  --output_dir output/yolo26nCls_siluInU8_attnS8
```

### Pose Smoke

```bash
pulsar2 build \
  --input yolo26_onnx/yolo26n_pose_qat_attnS8_smoke_slim.onnx \
  --config axera-npu/config_yolo26nPose_siluInU8_attnS8.json \
  --output_dir output_yolo26n_pose_qat_attnS8_smoke
```

### 其余任务

YOLO11n 检测使用 `axera-npu/config_yolo11n_qat.json`（或 `config_yolo11n_siluInU16_attnS8.json`），命令形态同检测模型。

具体 CLI 参数以客户工具链版本的 `pulsar2 build --help` 为准。

## 4. Python 推理与 COCO 结果导出

### 可用脚本

| 脚本 | 任务 | 输出 |
|---|---|---|
| `run_yolo_detect.py` | YOLO26/YOLO11 检测 | 检测 JSON |
| `run_yolo_seg.py` | 分割 | 检测与 mask JSON |
| `test.py --task obb` / `eval.py onnx-obb` | OBB 主机侧验证 | 旋转框 |
| `test.py --task pose` / `eval.py onnx-pose` | Pose 主机侧验证 | 检测框与关键点 |
| `test.py --task classify` | 分类主机侧验证 | 单个 `logits`（host 端 softmax → Top-K） |

两个脚本均从模型读取固定输入高宽，支持高宽可被最大 stride 32 整除的方形或矩形输入；例如 `352x640`。

OBB、Pose 和分类尚无独立 AXModel 推理脚本。

- **依赖**：这两个脚本只依赖 `numpy`、`opencv-python`、`tqdm` 和所选运行时；分割 JSON 导出额外需要 `pycocotools`。它们不依赖 `torch` 或 `ultralytics`，可作为 C++ 后处理的数值参考。
- **张量配对**：检测脚本按通道数和 anchor 数自动配对三个尺度的回归、分类张量，不依赖导出输出名称或顺序。它同时支持 YOLO26 的 4 通道距离回归和 YOLO11 的 `4 * reg_max` DFL 回归，后者在脚本内完成 softmax 期望值解码。

### 检测后处理契约

- 检测输出均为 `B,C,N`，按相同 `N` 配对三个尺度的 box 和 cls 张量。
- box 解码：YOLO26 的 box 为 `[1,4,N]` 直接距离；YOLO11 的 box 为 `[1,4*reg_max,N]`，需要 DFL 解码；cls 均需 sigmoid。
- 候选筛选：YOLO26 one-to-one 使用 top-k，YOLO26 one-to-many 与 YOLO11 使用类别感知 NMS。
- 若 one-to-many AXModel 不保留 `boxes_p*`/`scores_p*` 输出名，运行 `run_yolo_detect.py` 时必须传入 `--head-type one2many`。

### 运行时选择

两个脚本都有 `--runtime auto|onnxruntime|axengine`：

- `auto`：若已安装 `axengine`，使用 `AxEngineExecutionProvider`；否则使用 onnxruntime 的 CUDA/CPU provider。
- `onnxruntime`：用于 QuantONNX 基线核对，provider 从 `CUDAExecutionProvider`、`CPUExecutionProvider` 中选择。
- `axengine`：强制使用 `import axengine as ort` 和 `AxEngineExecutionProvider`；未安装或模型无法加载时直接报错。

### 检测示例

```bash
env PYTHONPATH="$PWD" \
  python axera-npu/run_yolo_detect.py \
  --runtime axengine --model output/yolo26n_qat_throughput/compiled.axmodel \
  --img-dir /path/to/coco/val2017 \
  --output-json output/yolo26n_qat_throughput_predictions.json
```

### 分割示例

```bash
env PYTHONPATH="$PWD" \
  python axera-npu/run_yolo_seg.py \
  --runtime axengine --model output/yolo26n_seg_qat/compiled.axmodel \
  --img-dir /path/to/coco/val2017 \
  --output-json output/yolo26n_seg_qat_predictions.json \
  --save-vis output/yolo26n_seg_qat_vis
```

用 `--runtime onnxruntime`、相同图片目录和阈值先跑 QuantONNX，再切换到 `axengine`，可直接比较
两份 JSON 的检测与分割精度。两种运行时都要求模型保留 `export.py` 的原始检测输出；不支持已插入
外部 NMS、改名或缺失检测/分割输出的模型。

## 5. 验收

1. 确认转换日志没有额外插入跨量化域 requant。
2. 使用客户统一的预处理、decode 和 COCO GT 评估 QuantONNX 与 AXModel。
3. 性能测试固定 warmup、repeat、输入尺寸和 NPU mode，例如：

```bash
ax_run_model -w 10 -r 100 -m output/yolo26n_qat_throughput/compiled.axmodel
```

4. 若 AXModel 精度异常，先对比 QuantONNX 和 NPUBackend 中间层，不要修改 QAT checkpoint 加载或手工
   扩大最终输出范围来掩盖问题。

## 6. 常见问题

- 节点名找不到：说明仍在使用历史 layer override，删除该 override，直接使用 QuantONNX Q/DQ。
- Split 分支精度下降：确认导出时未关闭 `--fix-split-reshape-quant`，并重新运行 skill 验收脚本。
- 出现 requant：确认 checkpoint 与配置 profile 一致，且使用当前 `export.py` 完整导出。
- 缺少 DequantizeLinear：升级到支持当前 ONNXScript/QuantONNX 的工具链版本。
- AxEngine provider 找不到：确认运行环境已安装 `axengine`，并使用 `--runtime axengine`；不要把
  `AxEngineExecutionProvider` 传给标准 onnxruntime。
