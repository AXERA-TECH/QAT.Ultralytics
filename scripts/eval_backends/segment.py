#!/usr/bin/env python3
"""Evaluate QAT segmentation checkpoint accuracy.

Usage:
    python eval.py segment --ckpt best.pt
    # Writes eval_seg_debug.log with full diagnostics.
"""

import argparse
import copy
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import torch

import ultralytics.utils.quantized_decomposed_dequantize_per_channel  # noqa: F401
from ultralytics import YOLO
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.qat_utils import prepare_pt2e_qat_model
from ultralytics.utils.torch_utils import select_device

warnings.filterwarnings("ignore")

DEBUG_LOG = Path.cwd() / "eval_seg_debug.log"


def log(msg: str):
    print(msg)
    with DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


class FakeTrainer:
    def __init__(self, float_model, qat_model, data_dict, device, end2end):
        self.model = float_model
        self.qat_model = qat_model
        self.device = device
        self.data = data_dict
        self.ema = None
        self.amp = False
        self.loss_items = torch.zeros(5)
        self.loss_names = ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss")
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
    parser = argparse.ArgumentParser(description="QAT seg eval")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--model", default="yolo26n-seg.yaml")
    parser.add_argument("--pretrained", default="./weights/yolo26n-seg.pt")
    parser.add_argument("--data", default="coco-seg.yaml")
    parser.add_argument("--quant-config", default="config-qat/config.json", dest="quant_config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)

    if DEBUG_LOG.exists():
        DEBUG_LOG.unlink()

    log("=" * 70)
    log("eval.py segment - yolo26n-seg QAT evaluation")
    log(f"ckpt:   {args.ckpt}")
    log(f"config: {args.quant_config}")
    log(f"device: {args.device}")
    log("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load float model
    # ------------------------------------------------------------------
    model = YOLO(args.model, task="segment").load(args.pretrained)
    float_model = model.model.float().to(device)
    float_model.train()

    hyp = dict(DEFAULT_CFG_DICT, **float_model.args) if isinstance(float_model.args, dict) else {}
    hyp.setdefault("box", 7.5)
    hyp.setdefault("cls", 0.5)
    hyp.setdefault("dfl", 1.5)
    hyp.setdefault("pose", 12.0)
    hyp.setdefault("kobj", 1.0)
    float_model.args = SimpleNamespace(**hyp)

    # ------------------------------------------------------------------
    # 2. Build PT2E prepared model (end2end=True for segmentation)
    # ------------------------------------------------------------------
    log("[step 2] prepare_pt2e_qat_model (end2end=True, segment) ...")
    _, prepared = prepare_pt2e_qat_model(
        float_model=float_model,
        device=device,
        config_path=args.quant_config,
        imgsz=args.imgsz,
        dynamic_batch_max=128,
    )

    # Cache criterion (E2ELoss + v8SegmentationLoss)
    float_model.criterion = float_model.init_criterion()
    log(f"  criterion cached: {type(float_model.criterion).__name__}")

    # ------------------------------------------------------------------
    # 3. Load QAT state
    # ------------------------------------------------------------------
    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    qat_state = ckpt.get("qat_ema") or ckpt.get("qat_model") or ckpt.get("model")
    if qat_state is None:
        log("ERROR: checkpoint has no 'qat_ema', 'qat_model' or 'model' key")
        sys.exit(1)
    log(f"  loading from {'qat_ema' if ckpt.get('qat_ema') is not None else 'qat_model'}")

    from ultralytics.engine.validator import BaseValidator

    prepared = BaseValidator._prepare_pt2e_model_for_eval(prepared)

    fresh_keys = set(prepared.state_dict().keys())
    ckpt_keys = set(qat_state.keys())
    log(f"  fresh prepared model keys: {len(fresh_keys)}")
    log(f"  checkpoint keys: {len(ckpt_keys)}")
    missing = fresh_keys - ckpt_keys
    unexpected = ckpt_keys - fresh_keys
    log(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")

    prepared.load_state_dict(qat_state)
    log("  strict=True: checkpoint matches the prepared QAT graph")
    # Evaluation must use the checkpoint's frozen qparams; updating observers on validation data
    # makes metrics batch-order dependent and diverges from trainer.validate().
    prepared.apply(torch.ao.quantization.disable_observer)
    prepared.to(device)
    log("[step 3] QAT state loaded")

    train_args = ckpt.get("train_args") or {}
    overlap_mask = bool(train_args.get("overlap_mask", True))
    mask_ratio = int(train_args.get("mask_ratio", 4))
    log(f"  segmentation data settings: overlap_mask={overlap_mask}, mask_ratio={mask_ratio}")

    # ------------------------------------------------------------------
    # 4. Read training reference
    # ------------------------------------------------------------------
    ckpt_epoch = ckpt.get("epoch", "?")
    results_csv = Path(args.ckpt).parent.parent / "results.csv"
    train_box, train_mask = "?", "?"
    if results_csv.exists():
        import csv

        with results_csv.open() as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i == ckpt_epoch:
                    train_box = float(row["metrics/mAP50-95(B)"])
                    train_mask = float(row["metrics/mAP50-95(M)"])
                    log(f"[ref] training epoch{ckpt_epoch}: box={train_box:.4f} mask={train_mask:.4f}")
                    break

    # ------------------------------------------------------------------
    # 5. Build validation dataloader
    # ------------------------------------------------------------------
    data_dict = check_det_dataset(args.data)
    gs = max(int(float_model.stride.max()), 32)

    val_bs = args.batch
    val_dataset = build_yolo_dataset(
        argparse.Namespace(
            task="segment",
            data=args.data,
            imgsz=args.imgsz,
            batch=val_bs,
            workers=args.workers,
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
            overlap_mask=overlap_mask,
            mask_ratio=mask_ratio,
            rect=False,
            cache=False,
        ),
        data_dict["val"],
        val_bs,
        data_dict,
        mode="val",
        rect=False,
        stride=gs,
    )
    val_loader = build_dataloader(
        val_dataset, batch=val_bs, workers=args.workers, shuffle=False, rank=-1, drop_last=False
    )

    # ------------------------------------------------------------------
    # 6. Validate
    # ------------------------------------------------------------------
    cfg = copy.deepcopy(DEFAULT_CFG_DICT)
    cfg.update(
        {
            "task": "segment",
            "mode": "val",
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": val_bs,
            "device": args.device,
            "workers": args.workers,
            "split": "val",
            "end2end": True,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "half": False,
            "plots": False,
            "save_json": False,
            "save_hybrid": False,
            "overlap_mask": overlap_mask,
            "mask_ratio": mask_ratio,
        }
    )

    validator = SegmentationValidator(dataloader=val_loader, args=cfg)
    fake_trainer = FakeTrainer(float_model, prepared, data_dict, device, True)

    log("\n[step 6] Running segment validation ...")
    import time as _time

    t0 = _time.time()
    results = validator(trainer=fake_trainer)
    elapsed = _time.time() - t0

    box_mAP50 = results.get("metrics/mAP50(B)", 0)
    box_mAP50_95 = results.get("metrics/mAP50-95(B)", 0)
    mask_mAP50 = results.get("metrics/mAP50(M)", 0)
    mask_mAP50_95 = results.get("metrics/mAP50-95(M)", 0)

    # ------------------------------------------------------------------
    # 7. Final comparison
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log(f"  eval.py segment epoch{ckpt_epoch}:")
    log(f"    box  mAP50:    {box_mAP50:.4f}")
    log(f"    box  mAP50-95: {box_mAP50_95:.4f}")
    log(f"    mask mAP50:    {mask_mAP50:.4f}")
    log(f"    mask mAP50-95: {mask_mAP50_95:.4f}")
    if train_box != "?":
        d_box = box_mAP50_95 - train_box
        d_mask = mask_mAP50_95 - train_mask
        log(f"  training box:  {train_box:.4f}  (delta={d_box:+.4f})")
        log(f"  training mask: {train_mask:.4f}  (delta={d_mask:+.4f})")
    log(f"  validation time: {elapsed:.0f}s")
    log("=" * 70)
    log(f"\nFull debug log saved to: {DEBUG_LOG}")


if __name__ == "__main__":
    main()
