---
name: qat-accuracy-attribution
description: 分阶段归因 YOLO PT2E QAT/PTQ 精度损失，并核查 `export_for_training` 图与 eager 模型的一致性。用于用户反馈 QAT 精度低于浮点或目标、需要判断损失在 backbone/neck/head、需要分段 fake-quant 消融（QAT/PTQ）、逐层误差或权重分布分析、怀疑导出图静默改变网络语义（如 C2f 列表突变追踪）时使用。
---

# QAT Accuracy Attribution

用于回答"量化损失在哪里"以及"损失是不是量化造成的"。核心纪律：**任何量化分析之前必须先通过 S0 图一致性检查**；图与 eager 不一致时先修模型代码并从浮点重训，否则后续所有结论无效。

## 环境与前置

- 在仓库根目录执行；激活项目兼容的 PyTorch 2.6 QAT 环境；设置 `PYTHONPATH="$PWD"`。
- 训练/评估前确认目标 GPU 空闲，不要打断其他训练。
- QAT checkpoint 与 quant config 的 observer 必须匹配：先用
  `$yolo-qat-config-discovery` 的 `--check` 验证配置命中。
- 报告与结论模板复用 `$qat-cn-todos` 的 `templates/analysis.md`；实验 JSON 写入 `cache/debug/`（`cache/` 为本地目录，不随项目提交）或任务目录。
- 全部分析在 CPU/单卡上进行；不修改 ultralytics 代码（除非 S0 判定需要修复模型代码）。

## 阶段工作流（按序执行）

### S0 图一致性（强制第一步）

```bash
python .codex/skills/qat-accuracy-attribution/scripts/check_export_graph_equivalence.py \
  --model yolov10s.yaml --pretrained weights/yolov10s.pt --device cuda:0
```

- 逐 `model.i` 比较 eager 层输出与 `torch.export.export_for_training` 图中对应节点输出；退出码 0 表示一致，1 表示分叉。
- 分叉定位：`--c2f-internal --c2f-layers <idx,...>` 深入 C2f 的 `cv1/m*/cv2`。
- 已知案例（2026-10）：`C2f.forward` 的 `y.extend(m(y[-1]) for m in self.m)` 被 dynamo 错误追踪，
  可变态列表突变后 `y[-1]` 被缓存为初值，C2f 内 bottleneck 退化为并行；修复方式是显式循环
  （`block.py` 的 `C2f.forward/forward_split`）。修复前 v10 从 layer 4 起分叉（diff 2.86），
  修复后全层 diff=0；26n/11n 图不受该写法影响。
- 处理规则：修复模型代码 → 重跑 S0 → **从浮点权重重训**（旧 checkpoint 的图语义已变化）；
  不得带着分叉图继续做任何量化归因。
- 模块级补充回归：`scripts/verify_block_fixes.py` 对生成器列表突变涉及的模块做
  eager（修改前后）与导出图一致性验证（要求 block 数 ≥2 才暴露该类问题）。

### S1 分段映射（backbone/neck/head）

```bash
python .codex/skills/qat-accuracy-attribution/scripts/qat_loss_attribution.py \
  --stage map --device cuda:0
```

- 输出 fake-quant 总数与分段落点、覆盖率与 unresolved 数量；unresolved 不应随数据集变化。

### S2 QAT 分段 fake-quant 消融（残余舍入）

```bash
python .codex/skills/qat-accuracy-attribution/scripts/qat_loss_attribution.py \
  --stage groups --device cuda:0 --batch 32
```

- 全量 COCO val 五组：`baseline / -backbone / -neck / -head / -all`，结果写 `cache/debug/attr_groups.json`。
- 语义：QAT 训练后逐段关闭 fake-quant 的残余舍入。
- 自检：`baseline` 应≈训练 `results.csv` 的最好值（±0.02）；`-all` 与 `baseline` 之差即为训练已吸收的舍入量。
- 判读：若全组差异 ≤0.2（噪声级），说明舍入不是损失来源，损失是量化约束下的容量损失，转向 S3/S6。

### S3 PTQ 分段消融（容量损失，无 QAT 补偿）

```bash
python .codex/skills/qat-accuracy-attribution/scripts/qat_loss_attribution.py \
  --stage ptq_groups --device cuda:0 --batch 32 --calib 500
```

- 浮点权重 + 校准后逐段关闭量化；结果写 `cache/debug/attr_ptq_ptq.json`。
- 自检：`ptq-all-fp32` 必须≈ eager `eval.py float`（否则先回 S0，图仍不一致）。
- 判读：某段关闭后恢复越大，该段容量损失越大；据此定位主因段。

### S4 数值偏差（QAT vs 浮点图）

```bash
python .codex/skills/qat-accuracy-attribution/scripts/qat_loss_attribution.py \
  --stage numeric --device cuda:0 --sample 100
```

- 输出 head 输入特征（neck 输出）与 per-scale boxes/scores 的 SNR/cosine。
- 注意：包含权重微调差异，只用于趋势参考，不单独下结论。

### S5 逐 conv 误差

```bash
python .codex/skills/qat-accuracy-attribution/scripts/qat_loss_attribution.py \
  --stage layerwise --device cuda:0 --sample 30
```

- 按执行顺序对齐两图逐 conv 输出，输出分段 SNR 中位数与最差层榜。
- 仅在 S0 PASS 后有意义；SNR 骤降位置提示误差累积点，与 S3 结论交叉验证。

### S6 权重分布对照

```bash
python .codex/skills/qat-accuracy-attribution/scripts/qat_loss_attribution.py --stage weights
```

- 统计各段 per-channel range/std，与 yolo26/yolo11 对照（yolo26 方法：range 越大越敏感）。
- 辅助解释，不单独作为归因。

## 判读表

| 观察 | 结论与下一步 |
|---|---|
| S0 非 0 | 导出图 bug：修模型代码 → 重验 S0 → 从浮点重训 |
| S2 各组差异 ≤0.2 | QAT 已吸收舍入；损失为容量损失，看 S3/S6 |
| S3 某段 fp32 恢复最大 | 该段容量损失主因；优先调整该段量化边界/位宽 |
| S5 某段 SNR 明显低于其他段 | 误差累积位置；用 S3 复核是否为主因 |
| S6 某段权重 range 显著偏大 | 该段量化敏感的先验证据（解释性） |

## 已知陷阱

- 手动取 dataloader 批次前向时必须 `.float()/255`（validator 内部会自动归一化）。
- 校准和前向 prepared 模型前必须走 `BaseValidator._prepare_pt2e_model_for_eval`，否则 BN 使用 batch stats。
- 分段禁用后必须自检 `-all`（或 `ptq-all-fp32`）与浮点/ eager 一致（max diff≈0），不一致说明禁用未生效。
- 训练图与单独导出图的节点名可能不同，逐层比较要按执行顺序对齐（S5 已实现）。
- smoke 必须使用 `imgsz=640`；小图 QAT 图与正式训练图不通用，指标也无参考意义。

## 严格约束

- 本 skill 仅用于归因实验；交付精度评估必须使用 `eval.py`（`$yolo26-qat-delivery`），不得用本 skill 的消融数字作为交付指标。
- 不得放宽 checkpoint 加载、修改 quant config 或手改 ONNX 来掩盖图分叉。
- 模型代码修复后必须：重跑 S0、跑 QAT 定向回归
  （`tests/test_pt2e_bn_patch.py`、`tests/test_qat_observer_config.py`、`tests/test_export_qat_prepare.py`、`tests/test_qat_export_postprocess.py`），
  并从浮点权重重新训练后再评估。
