#!/usr/bin/env python3
"""
PT2E PTQ (Post-Training Quantization) accuracy evaluation for yolo26n.

Chrom:
    1. Export float model -> prepare_pt2e (PTQ observers)
    2. Calibrate on COCO val subset
    3. convert_pt2e -> quantized model
    4. Validate on full COCO val through existing DetectionValidator

Usage:
    python eval.py ptq [--calib-samples 500] [--matmul-s8] [--end2end false]
"""

import argparse
import copy
import time
import warnings
from pathlib import Path

import torch
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
from torch.export import Dim

import ultralytics.utils.quantized_decomposed_dequantize_per_channel  # noqa: F401
from ultralytics import YOLO
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER
from ultralytics.utils.ax_quantizer import AXQuantizer, ax_load_config
from ultralytics.utils.torch_utils import select_device

warnings.filterwarnings("ignore", message=r"erase_node\(batch_norm_.*")
warnings.filterwarnings("ignore", message=r"enable_nhwc_conv")
warnings.filterwarnings("ignore", message=r".*AutoUpdate.*")


def parse_args():
    parser = argparse.ArgumentParser(description="PT2E PTQ yolo26n accuracy eval")
    parser.add_argument("--model", default="yolo26n.yaml")
    parser.add_argument("--pretrained", default="yolo26n.pt")
    parser.add_argument("--data", default="coco.yaml")
    parser.add_argument("--quant-config", default="config-qat/config.json", dest="quant_config")
    parser.add_argument("--matmul-s8", action="store_true", dest="matmul_s8")
    parser.add_argument("--calib-samples", type=int, default=500, dest="calib_samples")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--end2end", default="false")
    parser.add_argument("--name", default="ptq")
    return parser.parse_args()


class FakeTrainer:
    """Minimal trainer-like object for DetectionValidator training-mode path."""

    def __init__(self, float_model, quantized_model, data_dict, device, args_ns):
        self.model = float_model
        self.qat_model = quantized_model
        self.device = device
        self.data = data_dict
        self.ema = None
        self.amp = False
        self.loss_items = torch.zeros(3)
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        self.args = args_ns
        self.world_size = 1
        from collections import namedtuple

        self.stopper = namedtuple("Stopper", ["possible_stop"])(possible_stop=False)
        self.epoch = 999
        self.epochs = 1000

    def label_loss_items(self, loss_items=None, prefix="val"):
        if loss_items is not None:
            keys = [f"{prefix}/{x}" for x in self.loss_names]
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return [f"{prefix}/{x}" for x in self.loss_names]


def main():
    args = parse_args()
    end2end = args.end2end.lower() == "true"

    quant_config = "config-qat/config_matmul_s8.json" if args.matmul_s8 else args.quant_config
    device = select_device(args.device)

    LOGGER.info(f"PTQ: config={quant_config}, end2end={end2end}, calib={args.calib_samples}")

    # ---- 1. Load float model ----
    model = YOLO(args.model, task="detect").load(args.pretrained)
    float_model = model.model.float().to(device)
    float_model.model[-1].end2end = end2end

    # Fix model.args for loss computation (convert dict -> SimpleNamespace with required hyperparams)
    from types import SimpleNamespace

    hyp = dict(DEFAULT_CFG_DICT, **float_model.args) if isinstance(float_model.args, dict) else {}
    hyp.setdefault("box", 7.5)
    hyp.setdefault("cls", 0.5)
    hyp.setdefault("dfl", 1.5)
    hyp.setdefault("pose", 12.0)
    hyp.setdefault("kobj", 1.0)
    float_model.args = SimpleNamespace(**hyp)

    # ---- 2. Export + prepare_pt2e (PTQ mode) ----
    gc, rc = ax_load_config(quant_config, is_qat=False)
    quantizer = AXQuantizer()
    quantizer.set_global(gc)
    quantizer.set_regional(rc)

    inputs = torch.rand(1, 3, args.imgsz, args.imgsz, device=device)
    LOGGER.info("Exporting float model...")
    t0 = time.time()
    ep = torch.export.export_for_training(
        float_model,
        (inputs,),
        dynamic_shapes={"x": {0: Dim("batch", min=1, max=128), 2: Dim.AUTO, 3: Dim.AUTO}},
    )
    LOGGER.info(f"  export: {time.time() - t0:.1f}s")

    prepared = prepare_pt2e(ep.module(), quantizer)
    LOGGER.info(f"  prepare_pt2e: {time.time() - t0:.1f}s")

    try:
        torch.ao.quantization.allow_exported_model_train_eval(prepared)
    except Exception:
        pass

    # ---- 3. Calibrate ----
    data_dict = check_det_dataset(args.data)
    gs = max(int(float_model.stride.max()), 32)

    # Build a limited dataset for calibration (val split, no rect, first N samples)
    calib_dataset = build_yolo_dataset(
        argparse.Namespace(
            task="detect",
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
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
            overlap_mask=False,
            mask_ratio=4,
            rect=False,
            cache=False,
        ),
        data_dict["val"],
        args.batch,
        data_dict,
        mode="val",
        rect=False,
        stride=gs,
    )

    class LimitedDS(torch.utils.data.Subset):
        def __init__(self, ds, n):
            super().__init__(ds, range(min(n, len(ds))))

    calib_ds = LimitedDS(calib_dataset, args.calib_samples)
    calib_loader = torch.utils.data.DataLoader(
        calib_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=min(4, args.workers),
        pin_memory=True,
        collate_fn=getattr(calib_dataset, "collate_fn", None),
    )

    LOGGER.info(f"Calibrating on {len(calib_ds)} samples...")
    prepared.eval()
    t0 = time.time()
    with torch.no_grad():
        for batch in calib_loader:
            img = batch["img"].to(device).float() / 255.0
            _ = prepared(img)
    LOGGER.info(f"  done ({time.time() - t0:.1f}s)")

    # ---- 4. Convert ----
    LOGGER.info("Converting...")
    t0 = time.time()
    quantized = convert_pt2e(prepared)
    LOGGER.info(f"  convert_pt2e: {time.time() - t0:.1f}s")

    try:
        torch.ao.quantization.allow_exported_model_train_eval(quantized)
    except Exception:
        pass
    quantized.eval()
    quantized.to(device)

    # ---- 5. Validate ----
    # Build val dataset/dataloader the same way the trainer does
    val_dataset = build_yolo_dataset(
        argparse.Namespace(
            task="detect",
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
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
            overlap_mask=False,
            mask_ratio=4,
            rect=False,
            cache=False,
        ),
        data_dict["val"],
        args.batch,
        data_dict,
        mode="val",
        rect=False,
        stride=gs,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch=args.batch,
        workers=args.workers * 2,
        shuffle=False,
        rank=-1,
        drop_last=False,
    )

    LOGGER.info(f"Validating on {len(val_dataset)} images...")

    cfg = copy.deepcopy(DEFAULT_CFG_DICT)
    cfg.update(
        {
            "task": "detect",
            "mode": "val",
            "model": None,
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "split": "val",
            "end2end": end2end,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "half": False,
            "plots": False,
            "save_json": False,
            "save_hybrid": False,
            "name": args.name,
        }
    )
    validator = DetectionValidator(dataloader=val_loader, args=cfg)

    args_ns = argparse.Namespace(
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
    fake_trainer = FakeTrainer(float_model, quantized, data_dict, device, args_ns)

    t0 = time.time()
    results = validator(trainer=fake_trainer)
    elapsed = time.time() - t0

    mAP50 = results.get("metrics/mAP50(B)", 0)
    mAP50_95 = results.get("metrics/mAP50-95(B)", 0)

    print("\n" + "=" * 70)
    print(f"  PTQ Results  (end2end={end2end}, {Path(quant_config).stem})")
    print(f"  Calibration: {args.calib_samples} samples")
    print(f"  mAP50:       {mAP50:.4f}")
    print(f"  mAP50-95:    {mAP50_95:.4f}")
    print(f"  Val time:    {elapsed:.0f}s")
    print("=" * 70)

    out_path = Path(f"ptq_model_{Path(quant_config).stem}.pth")
    torch.save({"quantized_model": quantized.state_dict(), "mAP50_95": mAP50_95}, out_path)
    LOGGER.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
