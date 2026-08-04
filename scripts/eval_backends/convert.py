#!/usr/bin/env python3
"""Evaluate a QAT checkpoint after convert_pt2e using real Q/DQ operators."""

import argparse
import copy
import json
import warnings
from types import SimpleNamespace

import torch
from torch.ao.quantization.quantize_pt2e import convert_pt2e

import ultralytics.utils.quantized_decomposed_dequantize_per_channel  # noqa: F401
from ultralytics import YOLO
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.qat_utils import prepare_pt2e_qat_model
from ultralytics.utils.torch_utils import select_device

warnings.filterwarnings("ignore")


class FakeTrainer:
    def __init__(self, float_model, qat_model, data_dict, device, end2end, task):
        self.model = float_model
        self.qat_model = qat_model
        self.device = device
        self.data = data_dict
        self.ema = None
        self.amp = False
        self.loss_names = get_loss_names(float_model, task)
        self.loss_items = torch.zeros(len(self.loss_names))
        self.world_size = 1
        self.epoch = 0
        self.epochs = 1
        from collections import namedtuple

        self.stopper = namedtuple("S", ["possible_stop"])(possible_stop=False)
        self.args = argparse.Namespace(
            half=False,
            amp=False,
            compile=False,
            plots=False,
            end2end=end2end,
            conf=0.001,
            iou=0.7,
            max_det=300,
            single_cls=False,
            agnostic_nms=False,
            save_json=False,
            save_hybrid=False,
        )

    def label_loss_items(self, loss_items=None, prefix="val"):
        if loss_items is not None:
            return dict(zip([f"{prefix}/{x}" for x in self.loss_names], [round(float(x), 5) for x in loss_items]))
        return [f"{prefix}/{x}" for x in self.loss_names]


p = argparse.ArgumentParser(description="Evaluate a converted PT2E QAT detection model")
p.add_argument("--ckpt", required=True)
p.add_argument("--quant-config", dest="quant_config", required=True)
p.add_argument("--task", choices=("detect", "obb", "pose", "classify"), default="detect")
p.add_argument("--model", default="yolo26n.yaml")
p.add_argument("--pretrained", default="yolo26n.pt")
p.add_argument("--data", default="coco.yaml")
p.add_argument("--end2end", default="True")
p.add_argument("--rect", default="False")
p.add_argument("--pycoco", default="True")
p.add_argument("--device", default="cuda:0")
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--workers", type=int, default=4)
a = p.parse_args()
requested_e2e = a.end2end.lower() == "true"
rect = a.rect.lower() == "true"
pycoco = a.pycoco.lower() == "true"
device = select_device(a.device)


def get_loss_names(float_model, task):
    if task == "classify":
        return ("loss",)
    if task == "obb":
        return ("box_loss", "cls_loss", "dfl_loss", "angle_loss")
    if task == "pose":
        names = ("box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss")
        if getattr(float_model.model[-1], "flow_model", None) is not None:
            names += ("rle_loss",)
        return names
    return ("box_loss", "cls_loss", "dfl_loss")


def rebuild_task_model(model, task, model_yaml, data_dict):
    if task == "obb":
        model.model = model.task_map[task]["model"](model_yaml, nc=data_dict["nc"], ch=data_dict["channels"])
    elif task == "pose":
        model.model = model.task_map[task]["model"](
            model_yaml, nc=data_dict["nc"], ch=data_dict["channels"], data_kpt_shape=data_dict["kpt_shape"]
        )


def run_classify_convert(a, device):
    """Evaluate a converted (real Q/DQ) classification QAT checkpoint; metric is Top-1/Top-5."""
    from ultralytics.data.utils import check_cls_dataset
    from ultralytics.models.yolo.classify.val import ClassificationValidator

    data_dict = check_cls_dataset(a.data)
    m = YOLO(a.model, task="classify")
    # Match the checkpoint's dataset-sized classification head before preparing the PT2E graph.
    m.model = m.task_map["classify"]["model"](a.model, nc=data_dict["nc"], ch=3)
    m.load(a.pretrained)
    fm = m.model.float().to(device)
    fm.nc = data_dict["nc"]
    fm.names = data_dict["names"]
    fm.train()
    _, prepared = prepare_pt2e_qat_model(
        float_model=fm, device=device, config_path=a.quant_config, imgsz=a.imgsz, dynamic_batch_max=128
    )
    fm.criterion = fm.init_criterion()

    ckpt = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    qat_state = ckpt.get("qat_ema") or ckpt.get("qat_model") or ckpt.get("model")
    assert qat_state is not None, "checkpoint has no qat_ema/qat_model/model state"
    rep = prepared.load_state_dict(qat_state, strict=False)
    print(f"[load] missing={len(rep.missing_keys)} unexpected={len(rep.unexpected_keys)}", flush=True)
    assert not rep.missing_keys, f"observer mismatch: missing={len(rep.missing_keys)}"

    prepared.eval()
    converted = convert_pt2e(prepared).to(device)
    bn = sum(1 for n in converted.graph.nodes if "batch_norm" in str(n.target).lower())
    print(f"[convert] 真实量化模型，残留 BN={bn}", flush=True)

    cfg = copy.deepcopy(DEFAULT_CFG_DICT)
    cfg.update(
        {
            "task": "classify",
            "mode": "val",
            "data": a.data,
            "imgsz": a.imgsz,
            "batch": a.batch,
            "device": a.device,
            "workers": a.workers,
            "split": "val",
            "half": False,
            "plots": False,
            "save_json": False,
        }
    )
    validator = ClassificationValidator(args=cfg)
    validator.dataloader = validator.get_dataloader(data_dict["val"], a.batch)
    ft = FakeTrainer(fm, converted, data_dict, device, False, "classify")
    res = validator(trainer=ft)
    top1 = res.get("metrics/accuracy_top1", 0.0)
    top5 = res.get("metrics/accuracy_top5", 0.0)
    print(f"\n>>> convert 真实量化 classify  top1={top1:.4f}  top5={top5:.4f}", flush=True)
    print("EVAL_CONVERT_DONE", flush=True)


if a.task == "classify":
    run_classify_convert(a, device)
    raise SystemExit(0)

data_dict = check_det_dataset(a.data)
m = YOLO(a.model, task=a.task)
rebuild_task_model(m, a.task, a.model, data_dict)
m.load(a.pretrained)
fm = m.model.float().to(device)
fm.train()
fm.nc = data_dict["nc"]
fm.names = data_dict["names"]
fm.model[-1].end2end = requested_e2e
e2e = bool(fm.model[-1].end2end)
if e2e != requested_e2e:
    print(
        f"[convert] requested end2end={requested_e2e}, but {type(fm.model[-1]).__name__} has no one2one head; use end2end={e2e}",
        flush=True,
    )
model_args = getattr(fm, "args", {})
hyp = dict(DEFAULT_CFG_DICT, **model_args) if isinstance(model_args, dict) else {}
hyp.setdefault("box", 7.5)
hyp.setdefault("cls", 0.5)
hyp.setdefault("dfl", 1.5)
_, prepared = prepare_pt2e_qat_model(
    float_model=fm,
    device=device,
    config_path=a.quant_config,
    imgsz=a.imgsz,
    dynamic_batch_max=128,
)
fm.args = SimpleNamespace(**hyp)
fm.criterion = fm.init_criterion()

ckpt = torch.load(a.ckpt, weights_only=False, map_location="cpu")
qat_state = ckpt.get("qat_ema") or ckpt.get("qat_model") or ckpt.get("model")
assert qat_state is not None, "checkpoint has no qat_ema/qat_model/model state"
rep = prepared.load_state_dict(qat_state, strict=False)
print(f"[load] missing={len(rep.missing_keys)} unexpected={len(rep.unexpected_keys)}", flush=True)
assert not rep.missing_keys, f"observer mismatch: missing={len(rep.missing_keys)}"

# ★ convert_pt2e：训练内 fake-quant → 真实 Q/DQ
prepared.eval()
converted = convert_pt2e(prepared).to(device)
bn = sum(1 for n in converted.graph.nodes if "batch_norm" in str(n.target).lower())
print(f"[convert] 真实量化模型，残留 BN={bn}", flush=True)

gs = max(int(fm.stride.max()), 32)
ns = argparse.Namespace(
    task=a.task,
    data=a.data,
    imgsz=a.imgsz,
    batch=a.batch,
    workers=a.workers,
    fraction=1.0,
    augment=False,
    erasing=0.0,
    flipud=0.0,
    fliplr=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    mosaic=0.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    auto_augment=None,
    single_cls=False,
    classes=None,
    overlap_mask=False,
    mask_ratio=4,
    rect=rect,
    cache=False,
)
vds = build_yolo_dataset(ns, data_dict["val"], a.batch, data_dict, mode="val", rect=rect, stride=gs)
vl = build_dataloader(vds, batch=a.batch, workers=a.workers, shuffle=False, rank=-1, drop_last=False)
cfg = copy.deepcopy(DEFAULT_CFG_DICT)
cfg.update(
    {
        "task": a.task,
        "mode": "val",
        "data": a.data,
        "imgsz": a.imgsz,
        "batch": a.batch,
        "device": a.device,
        "workers": a.workers,
        "split": "val",
        "end2end": e2e,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "half": False,
        "plots": False,
        "save_json": pycoco,
        "save_hybrid": False,
    }
)
validator_class = PoseValidator if a.task == "pose" else OBBValidator if a.task == "obb" else DetectionValidator
validator = validator_class(dataloader=vl, args=cfg)
ft = FakeTrainer(fm, converted, data_dict, device, e2e, a.task)
res = validator(trainer=ft)
print(
    f"\n>>> convert 真实量化 mAP50-95={res.get('metrics/mAP50-95(B)', 0):.4f}  mAP50={res.get('metrics/mAP50(B)', 0):.4f}",
    flush=True,
)
if a.task == "pose":
    print(
        f">>> convert Pose mAP50-95={res.get('metrics/mAP50-95(P)', 0):.4f}  "
        f"mAP50={res.get('metrics/mAP50(P)', 0):.4f}",
        flush=True,
    )
if pycoco and getattr(validator, "jdict", None):
    validator.save_dir.mkdir(parents=True, exist_ok=True)
    predictions = validator.save_dir / "predictions.json"
    with predictions.open("w", encoding="utf-8") as f:
        json.dump(validator.jdict, f)
    coco_results = validator.eval_json(dict(res))
    print(f">>> pycocotools mAP50-95={coco_results.get('metrics/mAP50-95(B)', 0):.4f}", flush=True)
print("EVAL_CONVERT_DONE", flush=True)
