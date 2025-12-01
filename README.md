
工程基于Ultralytics仓库用于做yolo系列的QAT训练；

|model|map@50-95|map@50|
|--|--|--|
|yolov11s.pt|0.466|0.635|
|yolov11s_8w8f_qdq.onnx|0.456|0.628|

## 环境安装
基于官方工程，安装ultralytics库

```
pip install -r requirements.txt
```

安装额外库

```
pip install ultralytics
```

我们发现 `onnxruntime` 和 `onnxscript` 的其他版本可能引起精度误差和导出错误，因此**pytorch==2.6; onnxruntime==1.21.0 onnxscript==0.4.0** 是必须的。

## 数据集路径修改

修改 ./ultralytics/cfg/datasets/coco.yaml 中的数据集路径;

## QAT训练
```
python train.py
```

## onnx eval
```
python eval.py
```
eval精度如下：

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.456
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.628
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.495
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.286
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.498
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.633
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.354
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.591
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.645
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.463
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.698
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.810

```

## onnx test
```
python test.py
```
test会加载根目录下的bus.jpg文件进行推理，然后输出推理结果

<img src="./result.jpg" width="405px">
