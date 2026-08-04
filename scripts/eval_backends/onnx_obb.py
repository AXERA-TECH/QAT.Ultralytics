#!/usr/bin/env python3
"""Evaluate a three-output OBB QuantONNX model with ONNX Runtime."""

from __future__ import annotations

import argparse
import copy
import json
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.torch_utils import select_device


def parse_bool(value: str | bool) -> bool:
    """Parse a command-line boolean value."""
    return value if isinstance(value, bool) else value.lower() in {"true", "1", "yes"}


class OrtOBB(torch.nn.Module):
    """Wrap concatenated OBB ONNX outputs as a PT2E-style prediction dictionary."""

    def __init__(self, onnx_path, device, head, session=None):
        super().__init__()
        if session is None:
            import onnxruntime as ort

            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.device = device
        self.box_channels = 4 * int(getattr(head, "reg_max", 1))
        self.score_channels = int(head.nc)
        self.angle_channels = int(head.ne)
        self.strides = [int(value) for value in head.stride.tolist()]

        input_shape = session.get_inputs()[0].shape
        if len(input_shape) != 4 or not all(isinstance(value, int) for value in input_shape[-2:]):
            raise RuntimeError(f"OBB QuantONNX must have fixed NCHW spatial dimensions, got {input_shape}")
        self.input_hw = tuple(input_shape[-2:])

    def _classify_outputs(self, outputs):
        tensors = [torch.from_numpy(np.asarray(output)).to(self.device) for output in outputs]
        by_channels = {}
        output_names = [output.name for output in self.session.get_outputs()]
        if set(output_names) == {"boxes", "scores", "angle"}:
            by_channels = dict(zip(output_names, tensors))
            self._validate_outputs(by_channels)
            return by_channels

        expected = {self.box_channels: "boxes", self.score_channels: "scores", self.angle_channels: "angle"}
        if len(expected) != 3:
            raise RuntimeError(
                "Cannot identify OBB outputs by channels because box, score, and angle channel counts overlap"
            )
        for tensor in tensors:
            if tensor.ndim != 3:
                raise RuntimeError(f"Unexpected OBB QuantONNX output shape: {tuple(tensor.shape)}")
            kind = expected.get(tensor.shape[1])
            if kind is None or kind in by_channels:
                raise RuntimeError(f"Unexpected OBB QuantONNX output channels: {tuple(tensor.shape)}")
            by_channels[kind] = tensor
        if set(by_channels) != {"boxes", "scores", "angle"}:
            raise RuntimeError("OBB QuantONNX must provide concatenated boxes, scores, and angle outputs")
        self._validate_outputs(by_channels)
        return by_channels

    def _validate_outputs(self, outputs):
        expected_channels = {
            "boxes": self.box_channels,
            "scores": self.score_channels,
            "angle": self.angle_channels,
        }
        for name, tensor in outputs.items():
            if tensor.ndim != 3:
                raise RuntimeError(f"Unexpected OBB QuantONNX output shape: {tuple(tensor.shape)}")
            if tensor.shape[1] != expected_channels[name]:
                raise RuntimeError(
                    f"OBB QuantONNX output {name!r} has {tensor.shape[1]} channels; expected {expected_channels[name]}"
                )
        anchors = {tensor.shape[2] for tensor in outputs.values()}
        if len(anchors) != 1:
            raise RuntimeError("OBB QuantONNX outputs must use the same anchor count")

    def forward(self, x):
        if tuple(x.shape[-2:]) != self.input_hw:
            raise RuntimeError(
                f"OBB QuantONNX expects spatial input {self.input_hw}, got {tuple(x.shape[-2:])}; "
                "use matching --imgsz and disable rect validation"
            )
        per_image = [
            self._classify_outputs(self.session.run(None, {self.input_name: image.detach().cpu().numpy()}))
            for image in x.split(1)
        ]
        pred_dict = {
            key: torch.cat([outputs[key] for outputs in per_image], dim=0) for key in ("boxes", "scores", "angle")
        }
        batch = x.shape[0]
        pred_dict["feats"] = [
            torch.zeros(batch, 1, self.input_hw[0] // stride, self.input_hw[1] // stride, device=self.device)
            for stride in self.strides
        ]
        one2many = {
            key: [value.clone() for value in pred_dict[key]] if key == "feats" else pred_dict[key].clone()
            for key in pred_dict
        }
        return {"one2one": pred_dict, "one2many": one2many}


class FakeTrainer:
    """Provide the minimal trainer contract consumed by OBBValidator."""

    def __init__(self, float_model, qat_model, data, device):
        self.model = float_model
        self.qat_model = qat_model
        self.device = device
        self.data = data
        self.ema = None
        self.amp = False
        self.loss_items = torch.zeros(4, device=device)
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "angle_loss")
        self.world_size = 1
        self.epoch = 0
        self.epochs = 1
        self.stopper = namedtuple("Stopper", ["possible_stop"])(possible_stop=False)
        self.args = argparse.Namespace(
            half=False,
            amp=False,
            compile=False,
            plots=False,
            end2end=True,
            conf=0.001,
            iou=0.7,
            max_det=300,
            single_cls=False,
            agnostic_nms=False,
            save_json=False,
            save_hybrid=False,
        )

    def label_loss_items(self, loss_items=None, prefix="val"):
        if loss_items is None:
            return [f"{prefix}/{name}" for name in self.loss_names]
        return dict(zip((f"{prefix}/{name}" for name in self.loss_names), map(float, loss_items)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="Three-output OBB QuantONNX model.")
    parser.add_argument("--model", default="yolo26n-obb.yaml")
    parser.add_argument("--pretrained", default="weights/yolo26n-obb.pt")
    parser.add_argument("--data", default="dota8.yaml")
    parser.add_argument("--device", default="cpu", help="Device used for decode and metrics; ORT runs on CPU.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16, help="Dataloader batch; ORT still runs one image at a time.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rect", type=parse_bool, default=False)
    parser.add_argument("--save-json", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rect:
        raise ValueError("OBB QuantONNX has fixed spatial dimensions; --rect True is not supported")
    device = select_device(args.device)
    data = check_det_dataset(args.data)

    model = YOLO(args.model, task="obb")
    model.model = model.task_map["obb"]["model"](args.model, nc=data["nc"], ch=data["channels"])
    model.load(args.pretrained)
    float_model = model.model.float().to(device)
    float_model.nc = data["nc"]
    float_model.names = data["names"]
    float_model.model[-1].end2end = True
    model_args = getattr(float_model, "args", {})
    hyp = dict(DEFAULT_CFG_DICT, **model_args) if isinstance(model_args, dict) else {}
    hyp.setdefault("box", 7.5)
    hyp.setdefault("cls", 0.5)
    hyp.setdefault("dfl", 1.5)
    float_model.args = SimpleNamespace(**hyp)
    float_model.criterion = float_model.init_criterion()
    float_model.eval()

    wrapper = OrtOBB(args.onnx, device, float_model.model[-1])
    if wrapper.input_hw != (args.imgsz, args.imgsz):
        raise ValueError(f"--imgsz {args.imgsz} does not match QuantONNX input {wrapper.input_hw}")
    print(f"[ort] loaded OBB QuantONNX {args.onnx} (CPU EP, input={wrapper.input_hw})", flush=True)

    stride = max(int(float_model.stride.max()), 32)
    dataset_args = argparse.Namespace(
        task="obb",
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
    )
    dataset = build_yolo_dataset(dataset_args, data["val"], args.batch, data, mode="val", rect=False, stride=stride)
    dataloader = build_dataloader(
        dataset, batch=args.batch, workers=args.workers, shuffle=False, rank=-1, drop_last=False
    )

    validator_args = copy.deepcopy(DEFAULT_CFG_DICT)
    validator_args.update(
        task="obb",
        mode="val",
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        split="val",
        end2end=True,
        conf=0.001,
        iou=0.7,
        max_det=300,
        half=False,
        plots=False,
        save_json=args.save_json,
        save_hybrid=False,
    )
    validator = OBBValidator(dataloader=dataloader, args=validator_args)
    results = validator(trainer=FakeTrainer(float_model, wrapper, data, device))
    if args.save_json and validator.jdict:
        validator.save_dir.mkdir(parents=True, exist_ok=True)
        predictions = validator.save_dir / "predictions.json"
        with predictions.open("w", encoding="utf-8") as file:
            json.dump(validator.jdict, file)
        results = validator.eval_json(dict(results))
        print(f"[json] saved {len(validator.jdict)} predictions to {predictions}", flush=True)
    print(
        f"\n>>> OBB QuantONNX mAP50-95={results.get('metrics/mAP50-95(B)', 0):.4f} "
        f"mAP50={results.get('metrics/mAP50(B)', 0):.4f}",
        flush=True,
    )
    print("EVAL_ONNX_OBB_DONE", flush=True)


if __name__ == "__main__":
    main()
