#!/usr/bin/env python3
"""统一 QAT 模型 eval（替代 eval.py / eval_one2one.py / eval_headU16_official.py）。.

QAT 模型是 PT2E prepared graph（带 observer/fake-quant），不能走标准 model.val（observer 匹配 + fuse 都会出问题），
必须：prepare_pt2e_qat_model 重建同构图 → load ckpt 的 qat_model state（校验 missing==0）→ FakeTrainer + DetectionValidator。

口径（默认 = 官方对标口径）：
  --end2end True   one2one(NMS-free) / False one2many(+NMS)
  --rect     False square 640（validator 对 prepared 本就强制 square；官方口径）/ True 矩形
  --pycoco   True  额外跑 pycocotools/COCO（training 模式 validator 不自动触发，脚本手动写 predictions.json + eval_json）

用法：
  python eval.py qat --ckpt runs/detect/yolo26n-qat-throughput/weights/best.pt \
    --quant-config config-qat/config_siluInU8_attnS8_clsU16.json --device 0
  python eval.py qat --ckpt <o2m_best.pt> --quant-config <cfg> --end2end False
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # ordinal=nvidia-smi index，避免落到满载 GPU0
import argparse
import copy
import json
import warnings
from types import SimpleNamespace

import torch

import ultralytics.utils.quantized_decomposed_dequantize_per_channel  # noqa: F401
from ultralytics import YOLO
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.qat_utils import prepare_pt2e_qat_model
from ultralytics.utils.torch_utils import select_device

warnings.filterwarnings("ignore")


def _bool(x):
    return str(x).lower() in ("true", "1", "yes")


class FakeTrainer:
    """把 QAT prepared graph 喂给 DetectionValidator 的 training-val 路径（observer 天然匹配）。."""

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

        self.stopper = namedtuple("Stopper", ["possible_stop"])(possible_stop=False)
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


def parse_args():
    p = argparse.ArgumentParser(description="Unified QAT (PT2E) model eval")
    p.add_argument("--ckpt", required=True, help="QAT checkpoint (含 qat_model/qat_ema state)")
    p.add_argument("--quant-config", dest="quant_config", required=True, help="训练所用的 quant config json")
    p.add_argument("--task", choices=("detect", "obb", "pose", "classify"), default="detect")
    p.add_argument("--model", default="yolo26n.yaml")
    p.add_argument("--pretrained", default="yolo26n.pt")
    p.add_argument("--data", default="coco.yaml")
    p.add_argument("--end2end", default="True", type=_bool, help="True=one2one(NMS-free) / False=one2many")
    p.add_argument("--rect", default="False", type=_bool, help="False=square640(官方) / True=矩形")
    p.add_argument("--pycoco", default="True", type=_bool, help="额外跑 pycocotools/COCO")
    p.add_argument("--device", default="0", help="nvidia-smi 卡号（PCI order）")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--fake-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep fake quantization enabled (default). Disable only to diagnose PT2E graph alignment.",
    )
    return p.parse_args()


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


def run_classify(a, device):
    """Evaluate a classification QAT checkpoint (fake-quant) via ClassificationValidator.

    Classification has no boxes/NMS/end2end/pycoco; the metric is Top-1/Top-5. The QAT graph outputs a single logits
    tensor, which BaseValidator._rebuild_pt2e_predictions passes through unchanged for non-dict outputs.
    """
    from ultralytics.data.utils import check_cls_dataset
    from ultralytics.models.yolo.classify.val import ClassificationValidator

    data_dict = check_cls_dataset(a.data)
    model = YOLO(a.model, task="classify")
    # Classification fine-tunes the head to the dataset class count; rebuild so the prepared
    # graph (linear output dim) matches the checkpoint observers before loading QAT state.
    model.model = model.task_map["classify"]["model"](a.model, nc=data_dict["nc"], ch=3)
    model.load(a.pretrained)
    float_model = model.model.float().to(device)
    float_model.nc = data_dict["nc"]
    float_model.names = data_dict["names"]
    float_model.train()

    _, prepared = prepare_pt2e_qat_model(
        float_model=float_model, device=device, config_path=a.quant_config, imgsz=a.imgsz, dynamic_batch_max=128
    )
    float_model.criterion = float_model.init_criterion()
    prepared = BaseValidator._prepare_pt2e_model_for_eval(prepared)

    ckpt = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    qat_state = ckpt.get("qat_ema") or ckpt.get("qat_model") or ckpt.get("model")
    assert qat_state is not None, "ckpt 无 qat_ema/qat_model/model"
    src = (
        "qat_ema"
        if ckpt.get("qat_ema") is not None
        else ("qat_model" if ckpt.get("qat_model") is not None else "model")
    )
    rep = prepared.load_state_dict(qat_state, strict=False)
    print(f"[load] from {src}  missing={len(rep.missing_keys)}  unexpected={len(rep.unexpected_keys)}", flush=True)
    assert len(rep.missing_keys) == 0, f"❌ observer 不匹配 missing={len(rep.missing_keys)}（config 与训练不一致？）"
    prepared.apply(torch.ao.quantization.disable_observer)
    if not a.fake_quant:
        prepared.apply(torch.ao.quantization.disable_fake_quant)
    prepared.to(device)

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
    ft = FakeTrainer(float_model, prepared, data_dict, device, False, "classify")
    results = validator(trainer=ft)
    top1 = results.get("metrics/accuracy_top1", 0.0)
    top5 = results.get("metrics/accuracy_top5", 0.0)
    print("\n" + "=" * 60, flush=True)
    print(f">>> QAT classify  fake_quant={a.fake_quant}", flush=True)
    print(f"    ultralytics-metric  top1={top1:.4f}  top5={top5:.4f}", flush=True)
    print("=" * 60, flush=True)
    print("EVAL_QAT_DONE", flush=True)


def main():
    a = parse_args()
    device = select_device(a.device)
    if a.task == "classify":
        return run_classify(a, device)
    requested_tag = "one2one" if a.end2end else "one2many"
    print(
        f"[eval qat] requested={requested_tag}  rect={a.rect}  pycoco={a.pycoco}  ckpt={a.ckpt}  cfg={a.quant_config}",
        flush=True,
    )

    data_dict = check_det_dataset(a.data)
    # 1) float model → train() 图（forward_head 训练分支，供 export_for_training 追踪）
    model = YOLO(a.model, task=a.task)
    rebuild_task_model(model, a.task, a.model, data_dict)
    model.load(a.pretrained)
    float_model = model.model.float().to(device)
    float_model.nc = data_dict["nc"]
    float_model.names = data_dict["names"]
    float_model.train()
    # end2end 必须在 prepare 之前设：决定 export 追踪的图结构（one2one/one2many）与 criterion 类型。
    # ⚠️ ckpt 必须是同 end2end 训练的（e2eTrue ckpt 用 --end2end True，e2eFalse 用 --end2end False），否则 observer 不匹配。
    float_model.model[-1].end2end = a.end2end
    end2end = bool(float_model.model[-1].end2end)
    if end2end != a.end2end:
        print(
            f"[eval qat] requested end2end={a.end2end}, but {type(float_model.model[-1]).__name__} has no one2one head; use end2end={end2end}",
            flush=True,
        )
    tag = "one2one" if end2end else "one2many"
    model_args = getattr(float_model, "args", {})
    hyp = dict(DEFAULT_CFG_DICT, **model_args) if isinstance(model_args, dict) else {}
    hyp.setdefault("box", 7.5)
    hyp.setdefault("cls", 0.5)
    hyp.setdefault("dfl", 1.5)

    # 2) 重建同构 PT2E prepared graph（与 ckpt observer 匹配）
    _, prepared = prepare_pt2e_qat_model(
        float_model=float_model, device=device, config_path=a.quant_config, imgsz=a.imgsz, dynamic_batch_max=128
    )
    float_model.args = SimpleNamespace(**hyp)
    float_model.criterion = float_model.init_criterion()
    prepared = BaseValidator._prepare_pt2e_model_for_eval(prepared)

    # 3) load QAT state（校验匹配）
    ckpt = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    qat_state = ckpt.get("qat_ema") or ckpt.get("qat_model") or ckpt.get("model")
    assert qat_state is not None, "ckpt 无 qat_ema/qat_model/model"
    src = (
        "qat_ema"
        if ckpt.get("qat_ema") is not None
        else ("qat_model" if ckpt.get("qat_model") is not None else "model")
    )
    rep = prepared.load_state_dict(qat_state, strict=False)
    print(f"[load] from {src}  missing={len(rep.missing_keys)}  unexpected={len(rep.unexpected_keys)}", flush=True)
    assert len(rep.missing_keys) == 0, f"❌ observer 不匹配 missing={len(rep.missing_keys)}（config 与训练不一致？）"
    # 冻结 observer：评测期间 min/max 不得再被 val 数据更新（fake-quant 保持开启）
    prepared.apply(torch.ao.quantization.disable_observer)
    if not a.fake_quant:
        prepared.apply(torch.ao.quantization.disable_fake_quant)
    _oe = [m.observer_enabled for m in prepared.modules() if hasattr(m, "observer_enabled")]
    print(
        f"[freeze] disable_observer 应用于 {len(_oe)} 个 fake-quant，enabled 残留={sum(int(x) for x in _oe)} "
        f"fake_quant={a.fake_quant}",
        flush=True,
    )
    prepared.to(device)

    # 4) dataloader（rect 口径）
    gs = max(int(float_model.stride.max()), 32)
    val_ns = argparse.Namespace(
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
        rect=a.rect,
        cache=False,
    )
    val_dataset = build_yolo_dataset(val_ns, data_dict["val"], a.batch, data_dict, mode="val", rect=a.rect, stride=gs)
    val_loader = build_dataloader(
        val_dataset, batch=a.batch, workers=a.workers, shuffle=False, rank=-1, drop_last=False
    )

    # 5) validate
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
            "end2end": end2end,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "half": False,
            "plots": False,
            "save_json": a.pycoco,
            "save_hybrid": False,
        }
    )
    validator_class = PoseValidator if a.task == "pose" else OBBValidator if a.task == "obb" else DetectionValidator
    validator = validator_class(dataloader=val_loader, args=cfg)
    ft = FakeTrainer(float_model, prepared, data_dict, device, end2end, a.task)
    results = validator(trainer=ft)
    ul_map = results.get("metrics/mAP50-95(B)", 0.0)
    ul_map50 = results.get("metrics/mAP50(B)", 0.0)
    pose_map = results.get("metrics/mAP50-95(P)") if a.task == "pose" else None
    pose_map50 = results.get("metrics/mAP50(P)") if a.task == "pose" else None

    # 6) pycocotools（training 模式 validator 不自动触发 → 手动写 predictions.json + eval_json）
    coco_map = None
    if a.pycoco and getattr(validator, "jdict", None):
        validator.save_dir.mkdir(parents=True, exist_ok=True)
        pj = validator.save_dir / "predictions.json"
        with open(pj, "w") as f:
            json.dump(validator.jdict, f)
        print(f"[pycoco] {len(validator.jdict)} 条 → {pj}，eval_json ...", flush=True)
        try:
            st = validator.eval_json(dict(results))
            coco_map = st.get("metrics/mAP50-95(B)", None)
        except Exception:
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60, flush=True)
    print(f">>> QAT {tag}  rect={a.rect}", flush=True)
    print(f"    ultralytics-metric  mAP50-95={ul_map:.4f}  mAP50={ul_map50:.4f}", flush=True)
    if pose_map is not None:
        print(f"    pose-metric         mAP50-95={pose_map:.4f}  mAP50={pose_map50:.4f}", flush=True)
    if coco_map is not None:
        print(f"    pycocotools/COCO    mAP50-95={coco_map:.4f}", flush=True)
    print("=" * 60, flush=True)
    print("EVAL_QAT_DONE", flush=True)


if __name__ == "__main__":
    main()
