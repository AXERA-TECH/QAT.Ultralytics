# QAT Best 训练指南（单 GPU / 多 GPU）

本说明用于指导在本仓库中复现实验基线训练，包含：
- 单 GPU 训练命令（`debug_qat_train2017_tune4_e10`）
- 多 GPU 训练命令（`debug_qat_train2017_multigpu_tune4_e10_fixpy`）

## 1. 环境准备

推荐使用仓库调试环境：

```bash
conda activate qat.yolov5
```

安装依赖（首次使用时）：

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. 数据集准备

确保 `coco_train2017.yaml` 中的数据集路径配置正确，并且可以正常访问 `train2017/val2017` 与标注文件。

## 3. 单 GPU 训练（基线）

```bash
yolo detect train \
  model=yolo11n.yaml data=coco_train2017.yaml pretrained=yolo11n.pt \
  epochs=10 batch=64 imgsz=640 optimizer=SGD lr0=4e-5 lrf=0.2 \
  qat_enable_fake_quant_epoch=7 qat_disable_observer_epoch=9 qat_disable_fake_quant_epoch=-1 \
  qat_onnx_sp=./debug/exp_train2017_tune4.onnx qat_onnx_imgsz=[640,640] \
  device=[3] amp=False \
  project=runs/detect name=debug_qat_train2017_tune4_e10 exist_ok=True
```

## 4. 多 GPU 训练（DDP）

多进程训练时需显式设置 `PYTHONPATH`，避免子进程导入到 `site-packages` 中未打补丁的 `ultralytics`。

```bash
PYTHONPATH=/your_project_path/QAT.Ultralytics:$PYTHONPATH \
yolo detect train \
  model=yolo11n.yaml data=coco_train2017.yaml pretrained=yolo11n.pt \
  epochs=10 batch=64 imgsz=640 optimizer=SGD lr0=4e-5 lrf=0.2 \
  qat_enable_fake_quant_epoch=7 qat_disable_observer_epoch=9 qat_disable_fake_quant_epoch=-1 \
  qat_onnx_sp=./debug/exp_train2017_multigpu_tune4_fixpy.onnx qat_onnx_imgsz=[640,640] \
  device=[1,2] amp=False \
  project=runs/detect name=debug_qat_train2017_multigpu_tune4_e10_fixpy exist_ok=True
```

## 5. 训练产物

- 单 GPU 输出目录：`runs/detect/debug_qat_train2017_tune4_e10`
- 多 GPU 输出目录：`runs/detect/debug_qat_train2017_multigpu_tune4_e10_fixpy`
- 权重文件：各目录下 `weights/best.pt` 与 `weights/last.pt`
- 训练参数快照：各目录下 `args.yaml`
