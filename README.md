# QAT.Ultralytics

本仓库基于 Ultralytics，用于调试和验证 YOLO 系列模型的 QAT（Quantization Aware Training）训练、导出与部署转换流程。

## 项目目标

- 训练可用于部署的 QAT 模型。
- 验证导出的 ONNX/QDQ 模型精度。
- 为 AXModel 转换提供可复现的配置和排查方法。


## 精度参考

| model | map@50-95 | map@50 |
| -- | -- | -- |
| yolov11s-fp32 | 0.466 | 0.635 |
| yolov11s_8w8f_qdq.onnx | 0.456 | 0.628 |
| yolov11n-fp32 | 0.391 | 0.561 |
| yolov11n_8w8f_qdq.onnx | 0.383 | 0.542 |

## 环境要求

安装依赖：

```bash
pip install -r requirements.txt
pip install -e .
```

版本约束：

- `pytorch==2.6`
- `onnxruntime==1.21.0`
- `onnxscript==0.4.0`

如果使用 `yolo11n` 或自定义的轻量模型进行QAT，先阅读 [README_nano.md](./README_nano.md)。

## 快速开始

1. 确认数据集配置可用。默认脚本依赖 `coco.yaml`，需要保证其中的数据集路径正确。
2. 运行训练：

```bash
# 单卡
python train.py
# 多卡，nproc_per_node对应train-gpus.py中device数量
python -m torch.distributed.run --nproc_per_node 2 train-gpus.py

# 本仓库含自定义的qat参数，如果上述指令报错，可能使用了其他环境Ultralytics仓库，使用如下指令尝试
PYTHONPATH=/your/project/path/QAT.Ultralytics:$PYTHONPATH python train.py
PYTHONPATH=/your/project/path/QAT.Ultralytics:$PYTHONPATH python -m torch.distributed.run --nproc_per_node 2 train-gpus.py
```

3. 训练后评估 QAT 权重：

```bash
python eval.py
```

4. 导出 QuantONNX。运行前先确认 `export.py` 里的 `qat_weights` 指向本次训练输出：

```bash
python export.py
```

5. 使用示例图片推理验证：

```bash
python test.py
```
## 部署
请阅读 [qat_deployment.md](./compile/qat_deployment.md)。

