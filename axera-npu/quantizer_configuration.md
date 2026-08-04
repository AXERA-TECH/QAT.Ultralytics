# AXERA QAT Quantizer 配置说明

本文说明 `ultralytics/utils/ax_quantizer*.py` 在基础 PT2E quantizer 上增加的配置能力，以及修改
YOLO26/YOLO11 网络结构时需要注意的兼容边界。模型转换步骤见
[qat_deployment.md](./qat_deployment.md)。

## 1. Quantizer 文件分工

- `ultralytics/utils/ax_quantizer.py`：默认 PT2E QAT quantizer，负责读取 JSON 配置并构造 qspec。
- `ultralytics/utils/ax_quantizer_utils.py`：按 Aten 算子和融合子图标注 Conv、SiLU、MatMul、Concat、Split 等量化点。
- `ultralytics/utils/ax_quantizer_lsq.py`：LSQ 变体，不属于当前 `accuracy`/`throughput` 交付 profile。

Quantizer 按 `torch.export` 后的 Aten 算子和子图结构匹配，不依赖 YOLO YAML 中的层序号。当前交付
profile 中的 FX 节点名称限制位于 `config-qat/*.json`，不在 quantizer Python 代码中。

## 2. 新增配置能力

### 2.1 输入与输出量化解耦

算子可以分别配置输入和输出 dtype、量化范围：

```json
{
  "is_symmetric": true,
  "input": { "dtype": "S8", "qmin": -127, "qmax": 127 },
  "output": { "dtype": "U8", "qmin": 0, "qmax": 255 }
}
```

配置规则：

- 全局配置未填写 `output` 时，保持兼容行为，输出沿用输入 qspec。
- regional 配置未填写 `output` 时，只覆盖目标算子输入，输出保留全局配置或由下游输入推导。
- regional 配置明确填写 `output` 时，强制设置目标算子的输出 qspec。

该能力用于构造连续的 U16 或 S8 区域，以及在区域末端恢复 U8。

### 2.2 独立的输出对称性

`output_is_symmetric` 可以独立控制输出 qscheme：

```json
{
  "is_symmetric": false,
  "output_is_symmetric": true,
  "input": { "dtype": "U8", "qmin": 0, "qmax": 255 },
  "output": { "dtype": "S8", "qmin": -127, "qmax": 127 }
}
```

这允许算子输入使用 affine U8，输出切换到 symmetric S8，而不改变权重量化设置。

### 2.3 独立的输出 Observer

默认 quantizer 支持为输入和输出选择不同的激活 observer：

```json
{
  "act_observer": "moving_avg",
  "output_act_observer": "minmax"
}
```

未配置 `output_act_observer` 时，输出沿用 `act_observer`。当前交付配置统一使用
`moving_avg`，没有启用独立输出 observer。

### 2.4 Regional 输出量化

Regional 配置可以覆盖以下算子的输出 qspec：

- Conv、ConvTranspose
- Add、Mul 等双输入算子
- SiLU 等激活算子
- MatMul 等原有 regional 算子

因此可以通过配置表达 `U16 -> U16`、`S8 -> S8` 或 `S8 -> U8`，不需要在模型代码中插入
显式量化算子。

### 2.5 Conv 融合边界

当 Conv 后存在 BN 或可融合激活时，Conv regional output qspec 会放在匹配到的融合子图末端：

```text
Conv -> BN                 -> Q/DQ
Conv -> BN -> Activation   -> Q/DQ
```

不会把量化点插入 Conv 与 BN 之间。该逻辑按子图模式工作，对标准 YOLO26/YOLO11 以及保持相同
Aten 模式的自定义模块均适用。

### 2.6 Concat 共享量化参数

全局或 regional 配置可以启用：

```json
"share_qparam": true
```

启用后，Concat 输入和输出共享量化参数，避免各分支 observer 独立导致不必要的 requant，并使
Concat 这类数据搬运算子的量化域保持一致。

## 3. Attention S8 配置

Attention S8 是 YOLO26/YOLO11 可复用的基础策略：

- 每个 Attention 区域的两个 MatMul 输入均为 S8。
- 第一 MatMul 输出以及中间 Mul、Softmax 保持 S8。
- 第二 MatMul 输出恢复 U8。

- 不含分类头补偿的基础模板为 `config-qat/config_siluInU16_attnS8.json` 和 `config-qat/config_siluInU8_attnS8.json`。
- 当前 YOLO26 `accuracy`/`throughput` 在此基础上增加分类塔 U16，使用文件名带 `_clsU16` 的组合配置。
- YOLO11 不使用 YOLO26 的分类头 U16 区域；应从基础模板重新发现，生成其专用 JSON。
- quantizer 中没有写死 Attention 节点编号。

### 3.1 QAT 配置与已验证组合

所有配置都按各自 JSON 保持 Attention 所需的 S8 量化边界。`clsU16` 是当前 YOLO26n 分类头的可选
精度补偿，不应直接套用到 YOLO11、分割模型或结构已修改的模型：

| 配置                                                     | 普通 SiLU            | 头部局部配置                                                          | 建议用途                                                  |
| -------------------------------------------------------- | -------------------- | --------------------------------------------------------------------- | --------------------------------------------------------- |
| `config-qat/config_siluInU16_attnS8_clsU16.json`         | input U16，output U8 | 最后两个尺度的 SiLU、Conv 和 score output 为 U16                      | YOLO26n `accuracy` profile，优先保留激活精度              |
| `config-qat/config_siluInU8_attnS8_clsU16.json`          | input/output U8      | 最后两个尺度的 SiLU、Conv 和 score output 为 U16                      | YOLO26n `throughput` profile，优先降低带宽和计算开销      |
| `config-qat/config_siluInU8_attnS8_clsU16_one2many.json` | input/output U8      | `cv3` 最后两个尺度的 SiLU、Conv 和分类 logit 为 U16；最终 score 为 U8 | YOLO26n `end2end=false`，常规 NMS 路径                    |
| `config-qat/config_yolo26nSeg_siluInU16_attnS8.json`     | input U16，output U8 | 不启用 clsU16；mask coefficient 和 proto 保持全局量化                 | 当前 YOLO26n-seg `end2end=true` 推荐配置                  |
| `config-qat/config_yolo26nObb_siluInU8_attnS8.json`      | input/output U8      | 不修改                                                                | 当前 YOLO26n-OBB smoke 推荐配置，双 Attention S8          |
| `config-qat/config_yolo26nPose_siluInU8_attnS8.json`     | input/output U8      | 不修改                                                                | 当前 YOLO26n-Pose smoke 推荐配置，双 Attention S8         |
| `config-qat/config_yolo26nCls_siluInU8_attnS8.json`      | input/output U8      | 不修改                                                                | 当前 YOLO26n-Cls smoke 推荐配置，单 Attention S8          |
| `config-qat/config_yolo11n_siluInU8_attnS8.json`         | input/output U8      | 不修改                                                                | 当前 YOLO11n 检测图的推荐全 U8 配置                       |
| `config-qat/config_yolo11n_siluInU16_attnS8.json`        | input U16，output U8 | 不修改                                                                | 当前 YOLO11n 精度优先的已验证配置                         |
| `config-qat/config_siluInU16_attnS8.json`                | input U16，output U8 | 不修改                                                                | YOLO26 或自定义模型的 U16 基础模板，使用前需重新发现节点  |
| `config-qat/config_siluInU8_attnS8.json`                 | input/output U8      | 不修改                                                                | YOLO26 或自定义模型的全 U8 基础模板，使用前需重新发现节点 |

`train_qat.py` 的 `--profile accuracy|throughput` 仅对应前两项 YOLO26n 推荐配置；
`--quant-config <json>` 可指定任意配置，且优先于 `--profile`。YOLO11 必须显式使用专用 JSON：全 U8
`config_yolo11n_siluInU8_attnS8.json` 或精度优先的
`config_yolo11n_siluInU16_attnS8.json`；两者均保持单 Attention 连续 S8，且不启用 YOLO26 的 clsU16。

> **⚠️ 精度验证范围**
>
> 上表中**检测**（YOLO26n/YOLO11n）和**分割**（YOLO26n-seg）的量化配置已通过正式精度实验和上板评估，可作为交付基线。
> **OBB、Pose 和分类**的配置（`config_yolo26nObb_*`、`config_yolo26nPose_*`、`config_yolo26nCls_*`）仅完成主机侧链路 smoke，
> **未经正式精度调优，不保证在业务数据上满足精度要求**。使用前须在目标完整数据集上重新训练并评估，必要时通过
> `$yolo-qat-config-discovery` 重新发现局部量化边界，并经 AXERA 板端验证后才能作为交付结论。

`config-qat/` 是训练 JSON 的唯一存放目录；与其配对的 Pulsar2 转换 JSON 位于 `axera-npu/`，命名和生成规则见
[qat_deployment.md](./qat_deployment.md)。README 仅保留训练入口和本文链接，避免两处配置表在后续实验中漂移。

新增 OBB、Pose、分类或自定义网络任务时，不应直接套用上述检测配置的 regional 节点或导出 wrapper。先按
[$skill-yolo-qat-task-onboarding](../.codex/skills/yolo-qat-task-onboarding/SKILL.md) 确认 head、loss、
validator 与 QAT 输出契约，以 `config-qat/config.json` 完成基础 QAT smoke；仅在链路通过后，再使用
[$skill-yolo-qat-config-discovery](../.codex/skills/yolo-qat-config-discovery/SKILL.md) 为当前图生成 Attention
S8 或局部 U16 配置。

#### YOLO11n 后续优化建议

当前 `config_yolo11n_siluInU16_attnS8.json` 的 AX650N 结果为 `38.84/54.86`，相对 FP32 的
`39.40/55.30` 仍有 `-0.56/-0.44` 的量化损失。训练内 QAT 最佳 `38.92/54.67` 与上板结果接近，说明
部署转换不是主要损失来源；SiLU input U16 相比全 U8 QAT 有改善，但相对 PTQ 的 mAP50-95 净收益有限。

当前不再为该模型启动新的优化实验。后续若业务精度目标提高，应按以下顺序排查，而非直接复用 YOLO26 的
`clsU16` 节点配置：

1. 使用相同输入对齐浮点、QAT QuantONNX 与 AXModel 的 P3/P4/P5 原始 box、DFL 和 cls 输出，先定位损失在
   检测头还是 backbone/neck。
2. 若损失集中于某个头部张量，只对该模型重新发现的对应 Conv/SiLU/DFL 或 cls 局部边界配置 16bit，并重新
   验收 Attention S8、BN 和 requant；不能按历史 FX/ONNX 节点名复制 regional 配置。
3. 若各尺度量化误差均小而最终 mAP 仍有差距，检查 DFL decode、类别感知 NMS、阈值和 letterbox 是否与浮点
   评测完全一致，再考虑训练日程或 EMA 的影响。

### 3.2 YOLO26 one-to-many 差异

one-to-many 配置必须通过 `--branch cv3` 重新发现分类塔，不能复用 one-to-one 的 `one2one_cv3` 节点。
`clsU16` 作用于 Sigmoid 前的分类 logit；`export.py --end2end false` 的六个最终 score 输出为 U8，验收时
使用 `--skip-output-check`。

### 3.3 Attention 模块范围

**模块拓扑与数量**

- YOLO26 检测/OBB/Pose/分割 和 YOLO11 都在 backbone 第 10 层（`model.10`）使用 `C2PSA`；其 `PSABlock.Attention` 依次执行 `Q^T @ K` 和 `V @ attention` 两个 MatMul。
- YOLO26 还在 P5 的 `C3k2(..., attn=True)` 内使用一个 `PSABlock`（`model.22`）。
- 因此，YOLO26n 检测/OBB/Pose 当前有两个 Attention block、四个 MatMul；YOLO11n 当前有一个 Attention block、两个 MatMul。
- YOLO26n-cls 骨干较短（无 neck/FPN），`C2PSA` 位于 **第 9 层**（`model.9`），仅有**一个** Attention block、两个 MatMul；不含 P5 的第二 `C2PSA`。

**S8 边界与节点命名**

- 每个 Attention 的第一 MatMul output 为 S8；第二 MatMul output 回到 U8。
- 普通未启用 `attn=True` 的 `C3k2` 虽属于 CSP 类模块，但不包含 MatMul，不需要 `attnS8` regional 配置。
- 具体 FX/ONNX 节点名必须以 discovery 和导出图为准，不能在自定义网络中沿用 `matmul` 等历史名称。

**重新发现与导出约束**

- YOLO11 的其他规格、改造网络或导出环境变化后不能假设 MatMul 数量不变，必须重新发现节点。
- 标准 YOLO26 检测交付图还要求导出后 `BatchNormalization=0`、requant 节点为 0；导出阶段会对齐 Split 输入与相关 Reshape 分支的量化参数，以满足 AXERA 对 Split 输入输出量化参数一致的约束。
- 分割图可能存在独立的 mask-head requant，不能套用检测图的零 requant 断言。

## 4. 自定义 YOLO26/YOLO11 网络

修改通道数、重复次数，或者增加由现有 Conv、BN、SiLU、Concat、Split、MatMul 等支持算子组成的
模块，通常不会使 quantizer 本身失效。需要分别检查以下两类接口。

### 4.1 已知问题与当前处理结论

| 项目                              | 影响                                                                            | 当前处理结论                                                                  |
| --------------------------------- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Regional 配置使用 FX 节点名       | 修改网络后节点编号可能变化，原有局部 U16/S8 配置可能不再命中目标算子            | 暂不自动重映射节点，在本文说明限制；自定义网络导出后重新核对目标 Q/DQ dtype   |
| Regional 节点未命中时静默跳过     | 模型可能仍能导出，但目标算子回落到全局量化配置                                  | 暂时保留当前行为，在本文明确说明；不能只依据 ONNX 导出成功判断配置正确        |
| 新增 quantizer 未覆盖的自定义算子 | 可能保持浮点或产生额外 Q/DQ，AXERA 是否支持取决于实际导出图和工具链             | 不属于当前标准 YOLO26 交付范围，不在本次处理                                  |
| 检测头输出格式固定                | 改变输出数量、尺度或字典结构后，当前导出和测试 wrapper 可能不适用               | 在本文说明输出契约；自定义检测头需要同步适配导出、测试和评估入口              |
| Attention S8 与 Split/Reshape     | 当前标准图需要保持 Attention 连续 S8 量化域，并满足 AXERA 的 Split 量化参数约束 | 当前 `accuracy`/`throughput` 配置和导出后处理已经覆盖，标准交付图无需额外修改 |

以上结论只针对当前已验证的标准交付图。用户修改网络后，可以继续使用通用 quantizer 能力，但原
profile 中按名称选择的区域不能视为自动适配。

### 4.2 Regional 节点发现与验收

交付 profile 包含 `conv2d_*`、`silu__*`、`matmul_*`、`mul_*` 和 `softmax_*` 等 FX 节点名称。网络结构或
导出环境变化后，这些名称可能改变。regional 节点未命中时当前会回落到全局量化配置，因此“成功导出 ONNX”
不能证明局部 U16 或 Attention S8 已生效；必须同时核对 FX 命中和最终 ONNX Q/DQ dtype。

仓库提供 `$yolo-qat-config-discovery` 自动发现流程。它从 `torch.export` 图的模块来源和拓扑重新定位
所有 Attention 区域；使用带 `_clsU16` 的模板时，再定位最后两个分类尺度及分类入口 fan-out：

```bash
env PYTHONPATH="$PWD" \
  python .codex/skills/yolo-qat-config-discovery/scripts/discover_qat_config.py \
  --model yolo26n.yaml --pretrained yolo26n.pt \
  --base-config config-qat/config_siluInU8_attnS8.json \
  --output config-qat/config_custom_siluInU8_attnS8.json \
  --device cpu --imgsz 64
```

脚本根据模板自动判断是否启用 clsU16：基础模板只更新 Attention，文件名带 `_clsU16` 的组合模板
同时更新分类头。已知模型应有的 Attention 数量时，通过 `--expected-attention` 严格断言；当前
YOLO26n 交付图使用 2，YOLO11n 当前图使用 1。

### 4.3 检测头输出契约

当前 `export.py` 的检测导出 wrapper 约定：

- 模型返回 `one2one` 或 `one2many` 结果。
- 检测头包含三个尺度。
- 每个尺度分别提供 `boxes` 和 `scores`。

修改检测头返回格式、输出尺度数量或输出布局时，需要同步适配 `export.py`、`test.py` 和 ONNX
评估入口。仅修改 backbone/neck 且保持检测头输出契约不变时，不受此限制。

### 4.4 Split/Reshape 对齐

`export.py` 的 Split/Reshape 量化参数对齐按 ONNX 拓扑和量化范围匹配，不依赖固定节点名称。
自定义网络仍保留相同 Split fan-out 与 Reshape 结构时可以继续生效；拓扑不再匹配时会跳过。

## 5. LSQ 差异

LSQ quantizer 已支持独立 `output`、`output_is_symmetric` 和 `share_qparam`，但当前与默认
quantizer 仍有以下差异：

- 不支持独立的 `output_act_observer`。
- regional 未配置 `output` 时，仍默认输出沿用输入 qspec。

当前交付 profile 不使用 LSQ。启用 LSQ 前，需要重新检查配置语义、Q/DQ 结构和精度，不能直接
套用默认 quantizer 的已验证结论。

## 6. 导出验收

标准 YOLO26 检测 `accuracy`/`throughput` profile 导出后，使用交付验收脚本检查：

```bash
python \
  .codex/skills/yolo-qat-config-discovery/scripts/validate_qat_structure.py \
  yolo26_onnx/yolo26n_qat_throughput_slim.onnx \
  --quant-config config-qat/config_siluInU8_attnS8_clsU16.json \
  --ort --expect-aligned-split-reshape 2
```

标准检测交付图应满足：

- ONNX checker 和 ONNX Runtime 加载通过。
- BatchNormalization 数量为 0。
- `_requant` Identity 数量为 0。
- Attention MatMul 输入全部为 S8，输出为两个 S8、两个 U8。
- 两处 Split/Reshape 分支的量化参数完成对齐。

自定义网络可能改变 MatMul 或 Split 数量，不能直接套用固定数量断言，但仍应检查相同类型的量化边界，并以
实际 AXERA 工具链转换和精度结果作为最终验收。分割模型还应单独检查 mask-head requant。
