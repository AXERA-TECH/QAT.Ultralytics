#!/usr/bin/env python3
"""Evaluate a three-output Pose QuantONNX model with ONNX Runtime."""

from __future__ import annotations

import argparse
import copy
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.torch_utils import select_device
from scripts.eval_backends.onnx_utils import resolve_imgsz


class OrtPose(torch.nn.Module):
    """Wrap concatenated Pose ONNX outputs as a PT2E-style prediction dictionary."""

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
        self.keypoint_channels = int(head.nk)
        self.strides = [int(value) for value in head.stride.tolist()]

        input_shape = session.get_inputs()[0].shape
        if len(input_shape) != 4 or not all(isinstance(value, int) for value in input_shape[-2:]):
            raise RuntimeError(f"Pose QuantONNX must have fixed NCHW spatial dimensions, got {input_shape}")
        self.input_hw = tuple(input_shape[-2:])
        if any(self.input_hw[0] % stride or self.input_hw[1] % stride for stride in self.strides):
            raise RuntimeError(f"Pose QuantONNX input {self.input_hw} must be divisible by strides {self.strides}")
        self.anchor_count = sum(
            (self.input_hw[0] // stride) * (self.input_hw[1] // stride) for stride in self.strides
        )

    def _classify_outputs(self, outputs):
        tensors = [torch.from_numpy(np.asarray(output)).to(self.device) for output in outputs]
        output_names = [output.name for output in self.session.get_outputs()]
        if set(output_names) == {"boxes", "scores", "keypoints"}:
            classified = dict(zip(output_names, tensors))
        else:
            expected = {
                self.box_channels: "boxes",
                self.score_channels: "scores",
                self.keypoint_channels: "keypoints",
            }
            if len(expected) != 3:
                raise RuntimeError("Cannot identify Pose outputs because expected channel counts overlap")
            classified = {}
            for tensor in tensors:
                if tensor.ndim != 3:
                    raise RuntimeError(f"Unexpected Pose QuantONNX output shape: {tuple(tensor.shape)}")
                kind = expected.get(tensor.shape[1])
                if kind is None or kind in classified:
                    raise RuntimeError(f"Unexpected Pose QuantONNX output channels: {tuple(tensor.shape)}")
                classified[kind] = tensor
        self._validate_outputs(classified)
        return classified

    def _validate_outputs(self, outputs):
        expected_channels = {
            "boxes": self.box_channels,
            "scores": self.score_channels,
            "keypoints": self.keypoint_channels,
        }
        if set(outputs) != set(expected_channels):
            raise RuntimeError("Pose QuantONNX must provide concatenated boxes, scores, and keypoints outputs")
        for name, tensor in outputs.items():
            if tensor.ndim != 3 or tensor.shape[1] != expected_channels[name]:
                raise RuntimeError(
                    f"Pose QuantONNX output {name!r} has shape {tuple(tensor.shape)}; "
                    f"expected [B, {expected_channels[name]}, N]"
                )
        if len({tensor.shape[2] for tensor in outputs.values()}) != 1:
            raise RuntimeError("Pose QuantONNX outputs must use the same anchor count")
        anchors = next(iter(outputs.values())).shape[2]
        if anchors != self.anchor_count:
            raise RuntimeError(f"Pose QuantONNX expected {self.anchor_count} anchors, got {anchors}")

    def forward(self, x):
        if tuple(x.shape[-2:]) != self.input_hw:
            raise RuntimeError(f"Pose QuantONNX expects spatial input {self.input_hw}, got {tuple(x.shape[-2:])}")
        per_image = [
            self._classify_outputs(self.session.run(None, {self.input_name: image.detach().cpu().numpy()}))
            for image in x.split(1)
        ]
        pred_dict = {
            "boxes": torch.cat([outputs["boxes"] for outputs in per_image]),
            "scores": torch.cat([outputs["scores"] for outputs in per_image]),
            "kpts": torch.cat([outputs["keypoints"] for outputs in per_image]),
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
    """Provide the minimal trainer contract consumed by PoseValidator."""

    def __init__(self, float_model, qat_model, data, device):
        self.model = float_model
        self.qat_model = qat_model
        self.device = device
        self.data = data
        self.ema = None
        self.amp = False
        self.loss_names = ("box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss")
        if getattr(float_model.model[-1], "flow_model", None) is not None:
            self.loss_names += ("rle_loss",)
        self.loss_items = torch.zeros(len(self.loss_names), device=device)
        self.world_size = 1
        self.epoch = 0
        self.epochs = 1
        self.stopper = namedtuple("Stopper", ["possible_stop"])(possible_stop=False)
        self.args = argparse.Namespace(
            half=False, amp=False, compile=False, plots=False, end2end=True, conf=0.001, iou=0.7, max_det=300,
            single_cls=False, agnostic_nms=False, save_json=False, save_hybrid=False,
        )

    def label_loss_items(self, loss_items=None, prefix="val"):
        if loss_items is None:
            return [f"{prefix}/{name}" for name in self.loss_names]
        return dict(zip((f"{prefix}/{name}" for name in self.loss_names), map(float, loss_items)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--model", default="yolo26n-pose.yaml")
    parser.add_argument("--pretrained", default="weights/yolo26n-pose.pt")
    parser.add_argument("--data", default="coco8-pose.yaml")
    parser.add_argument("--device", default="cpu", help="Decode/metric device; ORT runs on CPU.")
    parser.add_argument("--imgsz", nargs="+", type=int, default=None, help="Optional ONNX input size: height width.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    data = check_det_dataset(args.data)
    model = YOLO(args.model, task="pose")
    model.model = model.task_map["pose"]["model"](
        args.model, nc=data["nc"], ch=data["channels"], data_kpt_shape=data["kpt_shape"]
    )
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

    wrapper = OrtPose(args.onnx, device, float_model.model[-1])
    input_hw = resolve_imgsz(args.imgsz, wrapper.input_hw)

    stride = max(int(float_model.stride.max()), 32)
    dataset_args = argparse.Namespace(
        task="pose", data=args.data, imgsz=input_hw, batch=args.batch, workers=args.workers, fraction=1.0,
        augment=False, erasing=0.0, flipud=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, mosaic=0.0, mixup=0.0,
        cutmix=0.0, copy_paste=0.0, auto_augment=None, single_cls=False, classes=None, overlap_mask=False,
        mask_ratio=4, rect=False, cache=False,
    )
    dataset = build_yolo_dataset(dataset_args, data["val"], args.batch, data, mode="val", rect=False, stride=stride)
    dataloader = build_dataloader(dataset, batch=args.batch, workers=args.workers, shuffle=False, rank=-1, drop_last=False)
    validator_args = copy.deepcopy(DEFAULT_CFG_DICT)
    validator_args.update(
        task="pose", mode="val", data=args.data, imgsz=input_hw[0], batch=args.batch, device=args.device,
        workers=args.workers, split="val", end2end=True, conf=0.001, iou=0.7, max_det=300, half=False,
        plots=False, save_json=False, save_hybrid=False,
    )
    validator = PoseValidator(dataloader=dataloader, args=validator_args)
    results = validator(trainer=FakeTrainer(float_model, wrapper, data, device))
    print(
        f"\n>>> Pose QuantONNX Box mAP50-95={results.get('metrics/mAP50-95(B)', 0):.4f} "
        f"Pose mAP50-95={results.get('metrics/mAP50-95(P)', 0):.4f}",
        flush=True,
    )
    print("EVAL_ONNX_POSE_DONE", flush=True)


if __name__ == "__main__":
    main()
