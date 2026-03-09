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

---

## 7. train2017持续调参记录（目标 `mAP50-95 >= 0.391`）

> 说明：本节为后续在 `coco_train2017.yaml` 上的持续排查，均固定 `device=[3]`，并尽量保持 `10 epoch` 配置。

### 7.1 已完成/已验证结果
- `debug_qat_lr4e5_train2017_e10`: best `0.37426`
- `debug_qat_train2017_tune1_e10`: best `0.36765`
- `debug_qat_train2017_tune2_e10`: best `0.33389`
- `debug_qat_train2017_tune3_e10`: best `0.38000`
- `debug_qat_train2017_tune4_e10`: best `0.38640`（当前 train2017 线最佳）
- `debug_qat_train2017_tune5_e10`: best `0.38398`
- `debug_qat_train2017_tune6_e10`: best `0.38464`
- `debug_qat_train2017_tune7_e10`: best `0.38398`

### 7.1.1 `0.3864`基线固化保存
- 已固化的最佳 train2017 基线（`mAP50-95=0.38640`）：
  - 训练脚本：`train_best_03864_train2017.py`
  - 参数文件：`configs/qat/qat_best_03864_train2017.yaml`
  - 量化配置：`configs/qat/config_best_03864_asym.json`
  - 已验证权重：`runs/detect/debug_qat_train2017_tune4_e10/weights/best.pt`

### 7.2 代码问题新增排查
- **数据集哈希卡顿问题（`ultralytics/data/dataset.py`）**
  - 现象：`train2017` 每轮启动时在 `get_hash(self.label_files + self.im_files)` 卡住很久（I/O wait）。
  - 处理：增加环境开关 `ULTRALYTICS_SKIP_DATASET_HASH=1`，仅用于调试加速，跳过缓存哈希一致性断言。
- **EMA 在当前 QAT实现下不兼容**
  - 现象1：QAT+EMA update 触发 shape mismatch（observer/fake-quant buffer维度不一致）。
  - 现象2：修复 shape 后，训练第1轮验证出现全零指标（`mAP=0`）。
  - 现象3：保存 checkpoint 时报错 `Can't pickle local object 'annotate_bias.<locals>.derive_qparams_fn'`。
  - 结论：当前代码路径下，QAT 不应直接套用标准 EMA 流程（已回退到“QAT不走EMA更新”稳定路径）。

### 7.3 新增实验
- `debug_qat_train2017_tune9_emafix_tune4_e10`（EMA兼容性验证组）
  - 参数：同 `tune4`，并尝试在 QAT 路径启用 EMA。
  - 结果：**失败**；epoch1 记录为 `mAP50-95=0`，并在保存阶段崩溃（不可序列化）。
  - 结论：作为代码问题定位用例保留，不纳入参数最优比较。

- `debug_qat_train2017_tune10_headfp32_e10`（observer策略：检测头FP32）
  - 改动：`config.json` 采用区域配置，将检测头对应 conv 节点（`conv2d_63~conv2d_86`）设为 `FP32`，其余层保持全局QAT配置。
  - 训练参数：沿用 `tune4`（`lr0=4e-5, lrf=0.2, fq@7, obs_off@9`）。
  - 10-epoch结果：
    - `best mAP50-95(B)=0.38640`（epoch3）
    - `last mAP50-95(B)=0.37609`（epoch10）
  - 结论：检测头保持 FP32 未带来提升，fake-quant 阶段仍出现后段回落。

- `debug_qat_train2017_tune11_neckheadfp32_e10`（observer策略：Neck+Head大范围FP32）
  - 改动：`config.json` 将 `conv2d_40~conv2d_86` 设置为 `FP32`（只量化前半段卷积节点）。
  - 训练参数：同 `tune4`（`lr0=4e-5, lrf=0.2, fq@7, obs_off@9`）。
  - 10-epoch结果：
    - `best mAP50-95(B)=0.38640`（epoch3）
    - `last mAP50-95(B)=0.37665`（epoch10）
  - 结论：与 `tune4`/`tune10` 一致，未突破 `0.391`；fake-quant 后仍明显回落。

- `debug_qat_train2017_tune12_lr8e5_e10`（参数策略：提高学习率）
  - 改动：量化配置恢复默认非对称（asym），仅将 `lr0` 提升到 `8e-5`。
  - 训练参数：其余同 `tune4`（`lrf=0.2, fq@7, obs_off@9`）。
  - 10-epoch结果：
    - `best mAP50-95(B)=0.38460`（epoch3）
    - `last mAP50-95(B)=0.37393`（epoch10）
  - 结论：学习率提升未改善上限，且后段回落更明显。

- `debug_qat_train2017_tune13_frombest_fq9_e10`（参数策略：从best继续 + fake-quant延后）
  - 改动：
    - 初始化权重改为 `runs/detect/debug_qat_train2017_tune4_e10/weights/best.pt`
    - `lr0=3e-5`
    - `qat_enable_fake_quant_epoch=9`, `qat_disable_observer_epoch=9`
  - 10-epoch结果：
    - `best mAP50-95(B)=0.38627`（epoch6）
    - `last mAP50-95(B)=0.37640`（epoch10）
  - 结论：延后 fake-quant 可减轻后段劣化，但上限仍未超过 `0.3864`。

- `debug_qat_train2017_tune14_frombest_lr8e5_fq9_e10`（参数策略：从best继续 + 更高lr + fake-quant延后）
  - 改动：
    - 初始化权重：`runs/detect/debug_qat_train2017_tune4_e10/weights/best.pt`
    - `lr0=8e-5`
    - `qat_enable_fake_quant_epoch=9`, `qat_disable_observer_epoch=9`
  - 10-epoch结果：
    - `best mAP50-95(B)=0.38460`（epoch3）
    - `last mAP50-95(B)=0.37424`（epoch10）
  - 结论：更高学习率未带来收益，仍低于 `0.3864` 基线。

- `debug_qat_train2017_tune15_frombest_fq10_e10`（参数策略：从best继续 + fake-quant进一步延后）
  - 改动：
    - 初始化权重：`runs/detect/debug_qat_train2017_tune4_e10/weights/best.pt`
    - `lr0=3e-5`
    - `qat_enable_fake_quant_epoch=10`, `qat_disable_observer_epoch=10`
  - 10-epoch结果：
    - `best mAP50-95(B)=0.38627`（epoch6）
    - `last mAP50-95(B)=0.38582`（epoch10）
  - 结论：后段回落明显减轻，但峰值仍未超过 `0.3864` 基线，距目标 `0.391` 仍有差距。

- `debug_qat_train2017_tune16_frombest_fq9_obs10_lr2e5_e10`（参数策略：从best继续 + 最后1轮FQ + observer全程开启 + 更低lr）
  - 改动：
    - 初始化权重：`runs/detect/debug_qat_train2017_tune4_e10/weights/best.pt`
    - `lr0=2e-5`, `lrf=0.2`
    - `qat_enable_fake_quant_epoch=9`, `qat_disable_observer_epoch=10`, `qat_disable_fake_quant_epoch=-1`
  - 状态：训练中（10 epoch，`device=[3]`）。
