# QAT Debug Log

## 2026-03-11

### 环境

- Python：激活项目环境后使用 `python`
- Python 版本: `3.10.19`
- Torch 版本: `2.6.0+cu126`
- 本轮验证主要在 CPU 上完成；CUDA/NVML 在当前会话里初始化有告警，但不影响本轮 PT2E 图对齐检查。

### 本轮代码调整

- 新增 `ultralytics/utils/pt2e_bn_patch.py`
    - monkey patch `torch.ao.quantization.pt2e.export_utils` 和 `qat_utils`
    - 保持 exported / prepared graph 中 BatchNorm 的 `momentum=0.03`、`eps=1e-3`
- 在 `ultralytics/utils/__init__.py` 启动时自动调用 `patch_pt2e_batchnorm_handling()`
- 清理 `ultralytics/engine/model.py`
    - 移除训练入口中的实验性 `export_for_training + onnx + exit()` 调试代码
    - 恢复正常 `self.trainer.train()` 主流程
    - 移除顶层 `onnx/onnxsim/onnxslim` 等可选依赖导入
- 调整 `ultralytics/engine/validator.py`
    - 新增 trainer 上 `export_model / export_model_eval / qat_model` 的 PT2E 评估分支
    - 对 exported/PT2E graph 先切 eval，再把原始输出重组为 validator 可消费的预测结构
- 调整 `ultralytics/engine/trainer.py`
    - 增加 `export_model_eval` 状态位
    - `read_results_csv()` 在无 `polars` 环境下回退到内置 `csv`，避免训练保存阶段无关失败
- 新增 `ultralytics/utils/qat_utils.py`
    - 封装 PT2E QAT 准备流程
    - 当前策略只保留 `N` 动态，固定 `H/W=imgsz`
- 调整 `ultralytics/engine/model.py`
    - 增加 `qat=True` 正式入口
    - 增加配置项：`qat`、`qat_config`、`qat_dynamic_batch_max`、`qat_validate`
- 调整 `ultralytics/engine/trainer.py`
    - `qat_model` 参与 optimizer / forward / grad clip
    - `qat_model` 写入 checkpoint 元信息
    - `qat_validate=False` 时跳过在线 validator 和 checkpoint-based final eval
- 新增 `tests/test_pt2e_bn_patch.py`
- 新增 `tests/test_qat_engine.py`
    - 增加 detect / segment 的 PT2E QAT 训练 + 在线验证 smoke test
    - 测试显式使用 `qat=True, qat_validate=True`
    - 标记为 `slow`

### 已完成验证

#### 1. 静态编译

命令：

```bash
python -m compileall ultralytics
```

结果：

- 通过

#### 2. PT2E BatchNorm 对齐检查

命令：手写 Python 脚本，分别验证：

1. `export_for_training()` 后切到 eval
2. `prepare_qat_pt2e()` 后关闭 observer/fake-quant 再切到 eval

结果：

- `exported_model`:
    - before: `momentum=0.03`, `eps=0.001`
    - after eval toggle: `training=False`, `momentum=0.03`, `eps=0.001`
- `prepared_model`:
    - before eval: `momentum=0.03`, `eps=0.001`
    - after eval: `momentum=0.03`, `eps=0.001`
- 数值对齐：
    - `prepared_vs_exported_eval_max_abs_diff = 2.384185791015625e-07`

结论：

- BN patch 生效
- `allow_exported_model_train_eval()` / `prepare_qat_pt2e()` 不再把 BN 超参漂移到 torch 默认值
- 关闭 observer/fake-quant 后，prepared model 与 exported eval model 已基本对齐

#### 3. 普通检测训练烟测

命令：

```bash
python - << 'PY'
from tests.test_engine import test_detect
test_detect()
PY
```

结果：

- `DetectionTrainer.train()` 主流程已恢复，可完成 1 epoch 训练、保存和 best checkpoint 验证
- 后续 predictor 子流程失败，原因是当前环境离线，无法下载 `yolo26n.pt`
- 失败与本轮 PT2E/QAT 改动无直接关系

#### 4. YOLO26 detect QAT 训练烟测

命令：

```bash
python - << 'PY'
from ultralytics import YOLO

model = YOLO('yolo26n.yaml')
metrics = model.train(
    data='coco8.yaml',
    imgsz=32,
    epochs=1,
    batch=2,
    device='cpu',
    workers=0,
    save=False,
    plots=False,
    qat=True,
)
print(type(metrics).__name__)
PY
```

结果：

- PT2E QAT graph 准备成功
- 1 epoch QAT 训练可跑完
- `Model.train()` 已返回 `DetMetrics` 对象

备注：

- 我在单独调试时执行过 `print(metrics)`，触发了 `DetMetrics.__str__` 的既有属性访问问题；这发生在训练成功返回之后，不是本轮 QAT 训练链路失败

#### 5. YOLO26 detect QAT 训练 + 在线验证烟测

命令：

```bash
python - << 'PY'
from ultralytics import YOLO

model = YOLO('yolo26n.yaml')
metrics = model.train(
    data='coco8.yaml',
    imgsz=32,
    epochs=1,
    batch=2,
    device='cpu',
    workers=0,
    save=False,
    plots=False,
    optimizer='SGD',
    lr0=1e-4,
    qat=True,
    qat_validate=True,
)
print(type(metrics).__name__)
PY
```

结果：

- 1 epoch detect QAT 训练完成
- 训练末尾 validator 成功跑完
- 返回类型为 `DetMetrics`

根因与修复：

- 之前的 reshape 报错不是 `Validator` 模式本身导致，而是 `val` dataloader 在 rect batching 下产出了 `64x64` 输入
- 当前 PT2E QAT 图固定导出为 `32x32`，因此进入 YOLO26 head 后在 reshape 处失败
- 修复方式：
    - `ultralytics/models/yolo/detect/train.py`
    - 当 `self.qat_model is not None` 且 `mode == "val"` 时禁用 rect batching
    - 让 val loader 与 PT2E 图保持固定 `imgsz`

#### 6. YOLO26 segment QAT 训练 + 在线验证烟测

命令：

```bash
python - << 'PY'
from ultralytics import YOLO

model = YOLO('yolo26n-seg.yaml')
metrics = model.train(
    data='coco8-seg.yaml',
    imgsz=32,
    epochs=1,
    batch=2,
    device='cpu',
    workers=0,
    save=False,
    plots=False,
    optimizer='SGD',
    lr0=1e-4,
    qat=True,
    qat_validate=True,
)
print(type(metrics).__name__)
PY
```

结果：

- 1 epoch segment QAT 训练完成
- 训练末尾 validator 成功跑完
- 返回类型为 `SegmentMetrics`

根因与修复：

- YOLO26 `Segment26` 在 training graph 下的 `proto` 为 `(mask_proto, semseg)` tuple
- `SegmentationValidator.postprocess()` 只接受 mask proto tensor
- 修复方式：
    - `ultralytics/engine/validator.py`
    - PT2E 预测重组时，`loss_preds` 继续保留完整 `raw_preds`
- 传给 validator 后处理的 `proto` 仅提取 tuple 第 0 项

#### 7. QAT smoke test 脚本化

新增文件：

- `tests/test_qat_engine.py`

覆盖内容：

- `test_detect_qat_validate_smoke`
- `test_segment_qat_validate_smoke`

说明：

- 两个测试都显式校验 `qat=True, qat_validate=True`
- 当前 `yolo26` 环境未安装 `pytest`，因此本轮未执行 `python -m pytest`
- 已用项目环境中的 `python` 直接调用测试函数完成等价验证
- `tests/test_qat_engine.py` 已通过 `py_compile`

### 历史问题与定位

#### 1. exported model validator dynamic shape 约束

尝试方式：

- 手工构造 `DetectionTrainer`
- `trainer._setup_train()`
- 用 `torch.export.export_for_training()` 导出 eager detection model
- 将导出图挂到 `trainer.export_model`
- 调用 `trainer.validator(trainer)`

观察到的问题：

1. 静态 batch 导出时，validator dataloader 的 batch 与导出样例 batch 不一致，触发 shape constraint error
2. 改成 dynamic batch 后，若同时把 `H/W` 声明成过宽动态范围，会触发 torch 2.6 的 constraint violation
3. torch 给出的建议是：

```text
Suggested fixes:
  H = 32
  W = 32
```

结论：

- 当前 validator 的 PT2E 分支已经接上，但正式 `prepare_qat` 入口必须使用更严格的 dynamic shape 策略
- 对 YOLO26 当前最合理的策略应是：
    - `N` 动态
    - `H/W` 先固定在实际训练尺度，或只允许满足 stride/当前图约束的离散范围

#### 2. detect validator reshape 错误已定位并修复

历史现象：

- `trainer.qat_model` 在 train batch 上可正常 forward/backward
- 在 validator 路径上会触发：

```text
RuntimeError: shape '[2, 2, 128, 1]' is invalid for input of size 2048
```

最终定位：

- 问题不是 `trainer.qat_model.eval()` 或 `smart_inference_mode()` 直接造成
- 实际是 rect val loader 产出的 batch 为 `64x64`
- 而固定空间尺寸 PT2E QAT 图只接受 `32x32`

当前状态：

- detect 在线验证已修复并通过烟测

### 当前判断

- 本轮“对齐”已经完成 detect + segment 的训练闭环验证：
    - BN patch 已接入并验证
    - 训练入口已从实验态恢复到可用态
    - validator 已具备 PT2E/exported model 接入点
- 已确认：
    - `yolo26` detect 支持 `qat=True, qat_validate=True`
    - `yolo26-seg` segment 支持 `qat=True, qat_validate=True`
- 当前默认 `qat_validate=False` 仍然保留，作为更保守的默认行为；需要在线验证时显式开启
- 剩余重点更偏向泛化与工程化，而不是 detect/segment 当前 smoke 链路本身

### 下一步建议

1. 把 detect/segment 已验证通过的 QAT validator 逻辑补成更明确的自动化测试
2. 评估是否将 `qat_validate` 默认值从 `False` 调整为 `True`，或继续保持显式开启策略
3. 继续检查 pose/obb 等其他任务是否也需要同样的 fixed-size val 策略
4. 视需要单独梳理 exported eval graph 与 training graph 的职责边界，减少 validator 分支复杂度
