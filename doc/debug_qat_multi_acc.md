# YOLO11n QAT Multi-GPU Debug Record

## 1. Branch
- branch: `qat_mnultigpu`

## 2. Experiment Goal
- Continue from the best `train2017` parameter set in `docs/debug_qat_acc.md` and verify multi-GPU QAT behavior.

## 3. Baseline Params (from `debug_qat_train2017_tune4_e10`)
- data: `coco_train2017.yaml`
- model: `yolo11n.yaml` + pretrained `yolo11n.pt`
- epochs: `10`
- batch: `64`
- imgsz: `640`
- optimizer: `SGD`
- lr0/lrf: `4e-5 / 0.2`
- augmentation: all disabled (`hsv/degrees/translate/scale/fliplr/mosaic/mixup=0`)
- qat schedule:
- `qat_enable_fake_quant_epoch=7`
- `qat_disable_observer_epoch=9`
- `qat_disable_fake_quant_epoch=-1`

## 4. Multi-GPU Experiment
- run name: `debug_qat_train2017_multigpu_tune4_e10_fixpy`
- device: `[1,2]`
- status: completed

## 5. Result
- Attempt 1 (failed before epoch start):
- Symptom: DDP child process imported `site-packages/ultralytics` instead of local repo code.
- Error: custom args `qat_enable_fake_quant_epoch/qat_disable_observer_epoch/qat_disable_fake_quant_epoch` reported as invalid.
- Root cause: multi-process environment `PYTHONPATH` did not prioritize `/home/heqi/project/QAT.Ultralytics`.
- Fix: rerun with `PYTHONPATH=/home/heqi/project/QAT.Ultralytics:$PYTHONPATH` to force DDP subprocesses to use local patched code.
- Attempt 2:
- status: completed (10 epoch, GPUs `1,2`)
- run dir: `runs/detect/debug_qat_train2017_multigpu_tune4_e10_fixpy`
- best `mAP50-95(B)`: `0.38984` (epoch `4`)
- last `mAP50-95(B)`: `0.38822` (epoch `10`)
- first epoch `mAP50-95(B)`: `0.38926`
- comparison to single-GPU best (`tune4=0.38640`): `+0.00344`
- gap to target `0.391`: `-0.00116`
