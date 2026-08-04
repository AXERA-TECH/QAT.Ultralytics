---
name: yolo-qat-task-onboarding
description: 为本仓库新增 YOLO 网络任务接入并验证 PT2E QAT 训练链路。用户需要为 OBB、Pose、分类、深度估计或自定义检测头新增 QAT 支持，排查新任务的 trainer/loss/head 输出不兼容，创建首个基础量化配置，或完成新任务 QAT smoke、导出和 AXERA 结构验收时使用。
---

# YOLO QAT 新任务接入

先证明浮点任务训练链路和 QAT 训练契约正确，再做量化精度优化。不要复用其他任务的 regional 节点编号，也不要通过放宽 checkpoint 加载、修改导出图或手改 ONNX 掩盖问题。

## 1. 确认任务契约

从目标 YAML 的最后一个模块确认实际 head 类，而不是依据文件名推断。`--task segment` 是训练器任务类型；
YOLO26 分割模型实际 head 是 `Segment26`。记录以下事实：

- `YOLO(..., task=...)` 是否能创建正确的 trainer、validator 与 `*Model`。
- `head.forward()` 在 `train()`、`eval()`、`export`、`end2end=True/False` 时的返回结构。
- `Model.loss(batch, preds)` 对应的 loss 类需要哪些字典键、尺度列表、辅助输出或 prototype。
- validator 和 predictor 对输出布局、任务专用标签的要求。

读取 `ultralytics/models/yolo/model.py` 的 task map、`ultralytics/nn/tasks.py` 的模型/损失选择、目标
head 和 `ultralytics/utils/loss.py`。对新 task 执行一次非 QAT 浮点训练或验证，不能把浮点任务故障归因于 QAT。

若任务尚未被 `train_qat.py` 接受，在 `--task` 的 choices 中增加标准 task 名；保持默认 detect/segment
行为不变。不要增加 `segment26` 之类的伪 task 名，模型 YAML 应决定 `Segment26`、`OBB26`、`Pose26` 等 head 变体。

## 2. 检查 QAT 训练数据流

`Model._prepare_qat_training()` 将训练态浮点模型交给 `prepare_pt2e_qat_model()`；训练循环直接执行：

```python
preds = trainer.qat_model(batch["img"])
loss, loss_items = unwrap_model(trainer.model).loss(batch, preds, teacher_preds=teacher_preds)
```

因此逐项核对：

1. PT2E 导出的训练态 graph 能处理固定空间尺寸和动态 batch。
2. QAT `preds` 与同一模型的浮点训练态 `preds` 在容器类型、字典键、tensor 数量及 shape 上兼容。
3. `loss` 在无 KD 时接受默认 `teacher_preds=None`；如新 loss 不支持 KD，显式拒绝或适配 KD，不能静默丢失 loss。
4. QAT validator 从 `trainer.qat_model` 取模型时仍使用该 task 的 validator，并能完成一个验证 batch。
5. DDP 下 QAT 模型由 trainer 包装；先单卡通过再验证至少两卡，避免在子进程外准备 PT2E 图。

遇到不兼容时，优先修复 head 的训练输出或 model/loss 的调用契约，并为该行为增加定向测试。不要为某一个任务改变 Detect、Segment、OBB 等既有输出的默认语义。

## 3. 建立基础量化配置

以 `config-qat/config.json` 为新任务的首个 smoke 配置，保持全局量化，不添加检测分类塔 U16、Attention S8
或固定 FX 节点的 regional 配置。模型是图像模型但结构发生变化时，仍须在目标 PyTorch、输入尺寸和
`end2end` 设置下重新构图。

以下情况再调用 `$yolo-qat-config-discovery`：

- 存在 Attention，需为完整 QKV/MatMul/Softmax 区域配置 S8。
- 已通过基础 QAT，但有可复现的特定分类/回归/掩码头精度损失，需要局部 U16。
- 网络、PyTorch 或导出环境变更，旧 regional 节点不再命中。

将任务配置放入 `config-qat/`，以模型和量化作用命名，不使用 `expNN`。配置生成后必须执行其 `--check`，不命中时不得开始正式训练。

## 4. 最小真实数据 Smoke

使用目标任务的真实格式数据集、浮点预训练权重和一张或少量图像运行 1 epoch；不要只运行 forward。
命令形态如下，替换 task、模型、权重、数据和 GPU：

```bash
env PYTHONPATH="$PWD" CUDA_DEVICE_ORDER=PCI_BUS_ID \
  python train_qat.py \
  --task obb --model yolo26-obb.yaml --pretrained weights/yolo26n-obb.pt \
  --data dota8.yaml --quant-config config-qat/config.json \
  --device 0 --epochs 1 --batch 1 --imgsz 640 --workers 0 \
  --fraction 1.0 --name task-qat-smoke --exist-ok
```

验收必须覆盖：PT2E prepare、至少一个反向传播 batch、任务 validator、`best.pt` 和 `last.pt` 保存。
单图 mAP 为零可以接受，只证明链路通过，不能作为精度结论。训练前检查数据 YAML、图片/标签路径及缓存；如
`--fraction` 与已有数据缓存行为不一致，使用单独的最小数据 YAML 或修复数据加载逻辑后再继续。

预训练权重的 head 类别数与 YAML 默认值不同（例如 DOTA OBB 权重为 15 类、YAML 默认 `nc=80`）时，必须在
加载权重前用数据集 `nc` 重建任务模型。训练日志必须明确核对完整加载数；YOLO26n-OBB 预期为
`Transferred 792/792 items from pretrained weights`。出现分类头 shape mismatch 或较小的加载数时，停止
QAT 训练，先修复模型重建/权重加载；此类问题会在关闭 fake-quant 时同样导致精度异常。

## 5. 导出与推理验收

确认训练 checkpoint 中存在 `qat_model`（若启用 EMA，另有 `qat_ema`），再扩展 `export.py` 支持该任务。
导出 wrapper 必须明确声明任务输出契约；不要把 detect 的 one2one/one2many 六输出 wrapper 套用于 OBB、
Pose、分类或分割。

导出后按任务验收：

- ONNX checker 与目标运行时可加载、可推理。
- 输入 shape、输出数量、layout、坐标或 mask/keypoint/angle 语义与浮点模型一致。
- QuantizeLinear/DequantizeLinear dtype、已知 requant、BN 融合和共享 Split/Reshape qparams 符合实际后端约束。
- 使用相同预处理和后处理，对一张固定图比较浮点、QAT PT2E 与 ONNX/AXERA 输出；先比较原始任务头输出，再比较 NMS 或任务后处理结果。

涉及 AXERA Pulsar2 转换时，使用 `$axera-quantonnx-config` 生成 layer config；它只处理 QuantONNX 的
Attention S8 定位，不代替本 skill 的任务输出验收。

### OBB 已验证契约

YOLO26 OBB head 为满足训练 loss，会拼接三个尺度的原始 box、class 和 angle；不要强制改成检测模型的六个
逐尺度输出。当前仓库的 OBB 闭环使用：

```bash
python eval.py qat --task obb --ckpt <best.pt> --quant-config <config.json> \
  --model yolo26n-obb.yaml --pretrained weights/yolo26n-obb.pt --data dota8.yaml
python eval.py convert --task obb --ckpt <best.pt> --quant-config <config.json> \
  --model yolo26n-obb.yaml --pretrained weights/yolo26n-obb.pt --data dota8.yaml
python export.py --task obb --model yolo26n-obb.yaml --pretrained weights/yolo26n-obb.pt \
  --qat-weights <best.pt> --quant-config <config.json> --out <model.onnx>
python test.py --task obb --model <best.pt-or-model_slim.onnx> \
  --model-yaml yolo26n-obb.yaml --pretrained weights/yolo26n-obb.pt --source <image>
```

OBB QuantONNX 输出为拼接后的 `boxes`、`scores`、`angle`。`test.py` 根据 OBB head stride 重建特征网格并完成
旋转框解码；训练、fake-quant、真实 Q/DQ 和 ONNX Runtime 必须在相同数据、输入尺寸和后处理阈值下对比。
当前只有 OBB 基础 Pulsar2 smoke 配置，没有 AXModel 后处理脚本；不能使用 `run_yolo_detect.py` 或检测的
Pulsar2 JSON 替代。

已验证的 DOTA8 smoke 使用全局 U8/S8、`batch=1`、`imgsz=640`、1 epoch，训练内
`mAP50-95/mAP50=0.856/0.995`；同条件浮点为 `0.891/0.995`。DOTA8 只有 4 张验证图，不能作为正式精度结论。

### Pose 已验证契约

YOLO26 Pose 训练态输出为 `boxes`、`scores`、`kpts`、`kpts_sigma` 和 `feats`。`kpts_sigma` 只供
`PoseLoss26` 的 RLE 分支使用，QuantONNX 仅导出 `boxes`、`scores`、`keypoints`：

```bash
python eval.py qat --task pose --ckpt <best.pt> --quant-config <config.json> \
  --model yolo26n-pose.yaml --pretrained weights/yolo26n-pose.pt --data <pose.yaml>
python eval.py convert --task pose --ckpt <best.pt> --quant-config <config.json> \
  --model yolo26n-pose.yaml --pretrained weights/yolo26n-pose.pt --data <pose.yaml>
python export.py --task pose --model yolo26n-pose.yaml --pretrained weights/yolo26n-pose.pt \
  --qat-weights <best.pt> --quant-config <config.json> --out <model.onnx>
python eval.py onnx-pose --onnx <model_slim.onnx> \
  --model yolo26n-pose.yaml --pretrained weights/yolo26n-pose.pt --data <pose.yaml>
```

`test.py --task pose` 会完成关键点解码和 letterbox 到原图的坐标缩放。当前没有 Pose Pulsar2 配置、AXModel
关键点后处理或板端精度评估脚本，不得复用检测或 OBB 后处理。

### Classify 已验证契约

分类 head `Classify` 与检测家族完全不同：train() 返回单个 `(b, nc)` logits tensor（非 dict、无
one2one/one2many），loss 为 `v8ClassificationLoss`（单项 `loss`，不接受 `teacher_preds`，classify 不启用 KD），
validator 为 `ClassificationValidator`（指标 Top-1/Top-5，无 mAP），数据用 `check_cls_dataset`（目录/数据集名，
无 `channels`/`kpt_shape`）。分类训练会把 head 重塑到数据集 `nc`。当前仓库闭环：

```bash
python eval.py qat --task classify --ckpt <best.pt> --quant-config <config.json> \
  --model yolo26n-cls.yaml --pretrained weights/yolo26n-cls.pt --data <cls-dataset> --imgsz 224
python eval.py convert --task classify --ckpt <best.pt> --quant-config <config.json> \
  --model yolo26n-cls.yaml --pretrained weights/yolo26n-cls.pt --data <cls-dataset> --imgsz 224
python export.py --task classify --model yolo26n-cls.yaml --pretrained weights/yolo26n-cls.pt \
  --qat-weights <best.pt> --quant-config <config.json> --out <model.onnx> --imgsz 224 224
python test.py --task classify --model <best.pt-or-model_slim.onnx> \
  --model-yaml yolo26n-cls.yaml --pretrained weights/yolo26n-cls.pt --data <cls-dataset> --source <image>
```

分类 QuantONNX 输出单个 `logits`：`ClassifyWrapper` **不追加 softmax**——softmax 不在 QAT 图内，追加会留下
未量化的 float 尾算子（前有 Q/DQ、后无 Q/DQ）；部署时 host 端做 softmax/argmax，`argmax(logits)==argmax(softmax)`。
`_rebuild_pt2e_predictions` 对非 dict 输出直接透传，因此分类无需检测的 box 解码路径。

分类骨干含一个 `C2PSA`，需用 `$yolo-qat-config-discovery` 以基础 attention S8 模板（`--cls-u16 off`、
`--expected-attention 1`）生成配置（如 `config-qat/config_yolo26nCls_siluInU8_attnS8.json`），否则 attention
的 MatMul/Softmax 会落到 S16 默认。`imagenet10` 链路 smoke 的导出图满足 `BN=0`、`_requant=0`、无 S16、
Attention 四点连续 S8、尾部收在 `logits` 的 DequantizeLinear。

注意：`Classify` head 的 `nn.Dropout(inplace=True)` 在训练图里是 `aten.dropout_.default`（in-place 变体）。
它已加入 `ax_quantizer_utils._is_share_obs_or_fq_op` 的 passive 共享列表，使 `pool→flatten→dropout→linear`
激活域连续共享，避免 Linear 输入产生多余的 U8→U8 requant。当前没有分类 Pulsar2 配置或 AXModel 后处理。

## 6. 回归、记录与交付边界

1. 为新任务至少增加 CLI、head/loss 输出契约或 smoke 的自动测试；已有 detect、segment 回归测试必须通过。
2. 如修改共享的 `trainer`、`loss`、`head` 或 quantizer，重新执行 detect、segment 的最小 QAT smoke。
3. 在同级 `todos-<task>/` 记录数据集、命令、量化配置、链路结论、精度结果和未解决问题；本地实验目录应受 `.gitignore` 管理。
4. 将可交付的任务范围、训练/导出命令和限制补入 `README.md` 与 `axera-npu/` 文档。文档使用仓库相对路径和占位路径，不写本机绝对路径。

只有当浮点基线、QAT smoke、任务输出验收和既有任务回归均通过后，才开始调优 regional 量化配置或启动正式精度实验。
