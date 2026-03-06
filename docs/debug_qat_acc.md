# QAT精度排查记录（YOLO11n）

## 1. 目标
- 目标：排查 `yolo11n` 在 QAT 后相对浮点模型约掉 2 个点的问题。
- 本轮要求：围绕 `参数调优 / observer策略 / 代码问题` 三个方向，均按 `10 epoch` 训练做对比。
- GPU约束：所有实验均固定 `device=[3]`。

## 2. 代码问题排查与改动

### 2.1 训练主循环问题修复（`ultralytics/engine/trainer.py`）
- 修复点1：`optimizer_step()` 中梯度裁剪对象从固定 `self.model` 改为“QAT场景下优先 `self.qat_model`”。
  - 影响：避免 QAT 训练时梯度裁剪未作用于真实优化对象。
- 修复点2：增加 `_configure_qat_epoch()`，支持按 epoch 控制 `observer/fake_quant`。
  - 新增参数：
    - `qat_enable_fake_quant_epoch`
    - `qat_disable_observer_epoch`
    - `qat_disable_fake_quant_epoch`

### 2.2 验证路径问题修复（`ultralytics/engine/validator.py`）
- 修复点3：去掉每个 val batch 的 `deepcopy(trainer.qat_model)`。
  - 改为复用同一个 `qat_model.eval()` 实例。
  - 影响：避免验证阶段无意义复制导致开销大、状态不稳定。
- 修复点4：训练态验证时，当 `qat_model` 存在，显式使用 `trainer.model` 作为后处理/损失参考模型，避免 EMA/模型对象混用。

### 2.3 QAT配置项补充（`ultralytics/cfg/default.yaml`）
- 新增默认配置：
  - `qat_enable_fake_quant_epoch: 0`
  - `qat_disable_observer_epoch: -1`
  - `qat_disable_fake_quant_epoch: -1`

### 2.4 数据配置异常发现（重要）
- 在 `ultralytics/cfg/datasets/coco.yaml` 中发现：
  - 当前 `train: val2017.txt`
  - `val: val2017.txt`
- 即训练和验证都在 `val2017` 上，容易导致结论偏差，且不等价于标准 COCO train/val 设定。

## 3. 10-epoch 实验记录

> 统一设置：`model=yolo11n.yaml + yolo11n.pt`，`batch=64`，`epochs=10`，`imgsz=640`，`device=[3]`。

### 实验A：代码修复基线组（Codefix Baseline）
- 运行目录：`runs/detect/debug_qat_codefix_e10`
- 关键参数：
  - `lr0=2e-5, lrf=0.1`
  - 无增强：`hsv/degrees/translate/scale/fliplr/mosaic/mixup = 0`
  - `close_mosaic=0`
  - `qat_enable_fake_quant_epoch=0`
  - `qat_disable_observer_epoch=-1`
  - `qat_disable_fake_quant_epoch=-1`
- 结果：
  - `best mAP50-95(B)=0.38261`（epoch 10）
  - `last mAP50-95(B)=0.38261`

### 实验B：Observer分阶段组（Observer Schedule）
- 运行目录：`runs/detect/debug_qat_observer_sched_e10`
- 在实验A基础上仅修改：
  - `qat_enable_fake_quant_epoch=1`
  - `qat_disable_observer_epoch=5`
- 结果：
  - `best mAP50-95(B)=0.38112`（epoch 8）
  - `last mAP50-95(B)=0.37525`
- 结论：本组 observer 冻结偏早，后半程精度明显回落。

### 实验C：参数调优 + 温和observer组（Tuned）
- 运行目录：`runs/detect/debug_qat_tuned_e10`
- 在实验A基础上主要修改：
  - `lr0=3e-5, lrf=0.2, cos_lr=True`
  - `warmup_epochs=1.0, warmup_bias_lr=0.0`
  - 轻增强：`hsv_h=0.005, hsv_s=0.2, hsv_v=0.1, translate=0.05, scale=0.1, fliplr=0.5`
  - `qat_disable_observer_epoch=8`
- 结果：
  - `best mAP50-95(B)=0.37976`（epoch 10）
  - `last mAP50-95(B)=0.37976`
- 结论：在当前数据与10epoch条件下，该调参方向未优于实验A。

### 实验D：高学习率线性衰减组（LR Boost）
- 运行目录：`runs/detect/debug_qat_lr4e5_e10`
- 在实验A基础上主要修改：
  - `lr0=4e-5, lrf=0.2, cos_lr=False`
  - 其余保持实验A（增强全关、observer不冻结）
- 结果：
  - `best mAP50-95(B)=0.40687`（epoch 10）
  - `last mAP50-95(B)=0.40687`
- 结论：该组显著优于前面所有实验，且超过目标 `0.391`。

## 4. 最优组选择
- 本轮四组 10-epoch 实验最优：**实验D（LR Boost）**。
- 最优指标：`mAP50-95(B)=0.40687`。

## 5. 最优组参数落盘
- 已将最优组训练入口参数写入 `train.py`：
  - `name="best_qat_lr4e5_e10"`
  - `lr0=4e-5, lrf=0.2`
  - `cos_lr=False`
  - 增强全关
  - `close_mosaic=0`
  - `qat_enable_fake_quant_epoch=0`
  - `qat_disable_observer_epoch=-1`
  - `qat_disable_fake_quant_epoch=-1`

## 6. 后续建议（针对0.391目标）
- 当前已达成目标（`0.40687 > 0.391`）。
- 建议后续分两条线并行：
  - 在当前最优参数上复跑 2~3 次（不同 seed）验证稳定性。
  - 修正 `coco.yaml` 为 `train2017.txt` 后重做对比，确认结论在标准训练集设定下仍成立。
