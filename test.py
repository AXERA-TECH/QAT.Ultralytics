#!/usr/bin/env python3
"""Run YOLO QAT checkpoint or QuantONNX inference and draw detection, segmentation, or OBB results."""

from __future__ import annotations

import argparse
import glob
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.engine.results import Results
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import nms, ops
from ultralytics.utils.files import increment_path
from ultralytics.utils.qat_utils import prepare_pt2e_qat_model, resolve_qat_config_path
from ultralytics.utils.torch_utils import select_device
import ultralytics.utils.quantized_decomposed_dequantize_per_channel  # noqa: F401


DEFAULT_QAT_MODEL = "runs/detect/exp58-globalSiluU8AttnS8-e2eTrue-noEMA/weights/best.pt"
TASK_DEFAULTS = {
    "detect": ("yolo26n.yaml", "yolo26n.pt"),
    "segment": ("yolo26n-seg.yaml", "weights/yolo26n-seg.pt"),
    "obb": ("yolo26n-obb.yaml", "weights/yolo26n-obb.pt"),
    "pose": ("yolo26n-pose.yaml", "weights/yolo26n-pose.pt"),
    "classify": ("yolo26n-cls.yaml", "weights/yolo26n-cls.pt"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test YOLO QAT checkpoints or exported QuantONNX models.")
    parser.add_argument("--model", default=DEFAULT_QAT_MODEL, help="QAT best.pt or exported QuantONNX .onnx model.")
    parser.add_argument(
        "--source", default="ultralytics/assets/bus.jpg", help="Input image, image directory, or image glob."
    )
    parser.add_argument("--task", choices=sorted(TASK_DEFAULTS), default="detect", help="Model task.")
    parser.add_argument("--quant-config", default=None, help="QAT config override; defaults to checkpoint train_args.")
    parser.add_argument("--model-yaml", default=None, help="Reference model architecture; defaults by --task.")
    parser.add_argument("--pretrained", default=None, help="Reference float checkpoint; defaults by --task.")
    parser.add_argument("--data", default=None, help="Classification dataset for class names (classify task only).")
    parser.add_argument("--device", default="cpu", help="PT2E and decode device, for example cpu, 0, or cuda:0.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square model input size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--project", default="runs/predict", help="Output project directory.")
    parser.add_argument("--name", default="qat-test", help="Output run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse the output directory.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO-format detection labels.")
    parser.add_argument("--save-conf", action="store_true", help="Include confidence in saved labels.")
    parser.add_argument("--line-width", type=int, default=None, help="Bounding-box line width.")
    return parser.parse_args()


def as_bool(value) -> bool:
    return value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes"}


def find_images(source: str) -> list[Path]:
    path = Path(source)
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(item for item in path.rglob("*") if item.is_file())
    elif any(character in source for character in "*?[]"):
        candidates = [Path(item) for item in sorted(glob.glob(source, recursive=True))]
    else:
        raise FileNotFoundError(f"Input source not found: {source}")

    images = [item for item in candidates if item.suffix[1:].lower() in IMG_FORMATS]
    if not images:
        raise RuntimeError(f"No supported images found in source: {source}")
    return images


def build_reference_model(model_yaml: str, pretrained: str, device: torch.device, task: str):
    pretrained_names = YOLO(pretrained, task=task).names
    model = YOLO(model_yaml, task=task)
    if task in {"obb", "pose"}:
        # Task YAML defaults may differ from the checkpoint head metadata.
        kwargs = {"nc": len(pretrained_names), "ch": 3}
        if task == "pose":
            kwargs["data_kpt_shape"] = YOLO(pretrained, task=task).model.model[-1].kpt_shape
        model.model = model.task_map[task]["model"](model_yaml, **kwargs)
    model.load(pretrained)
    float_model = model.model.float().to(device)
    float_model.names = pretrained_names
    float_model.model[-1].end2end = True
    return float_model


@dataclass
class InferenceOutput:
    predictions: torch.Tensor
    proto: torch.Tensor | None = None


def decode_predictions(raw_predictions, reference_model, task: str) -> InferenceOutput:
    predictions, _ = BaseValidator._rebuild_pt2e_predictions(raw_predictions, reference_model)
    decoded = predictions[0] if isinstance(predictions, tuple) else predictions
    if task == "segment":
        if not isinstance(decoded, tuple) or len(decoded) != 2:
            raise RuntimeError("Segmentation predictions must contain decoded detections and mask prototypes")
        inference, proto = decoded
    else:
        inference, proto = decoded, None
    if not reference_model.model[-1].end2end:
        # Conventional YOLO heads return [B, 4 + nc (+ nm), N]. Convert to the common detection layout used below.
        inference = nms.non_max_suppression(
            inference,
            conf_thres=0.001,
            iou_thres=0.7,
            max_det=300,
            rotated=task == "obb",
        )[0].unsqueeze(0)
    head = reference_model.model[-1]
    expected_columns = (
        7 if task == "obb" else 6 + int(head.nk) if task == "pose" else 6 if task == "detect" else 6 + int(head.nm)
    )
    if inference.ndim != 3 or inference.shape[-1] != expected_columns:
        raise RuntimeError(f"Unexpected decoded prediction shape: {tuple(inference.shape)}")
    return InferenceOutput(inference, proto)


class QATCheckpointBackend:
    def __init__(self, args: argparse.Namespace, reference_model, device: torch.device):
        checkpoint = torch.load(args.model, weights_only=False, map_location="cpu")
        train_args = checkpoint.get("train_args") or {}

        config_path = resolve_qat_config_path(args.quant_config or train_args.get("qat_config", ""))
        if not config_path.is_file():
            raise FileNotFoundError(f"QAT config not found: {config_path}")

        reference_model.train()
        _, prepared = prepare_pt2e_qat_model(
            float_model=reference_model,
            device=device,
            config_path=str(config_path),
            imgsz=args.imgsz,
            dynamic_batch_max=int(train_args.get("qat_dynamic_batch_max", 128)),
        )
        prepared = BaseValidator._prepare_pt2e_model_for_eval(prepared)

        qat_state = checkpoint.get("qat_ema") or checkpoint.get("qat_model")
        if qat_state is None:
            raise KeyError("Checkpoint has no qat_ema or qat_model state")
        load_result = prepared.load_state_dict(qat_state, strict=False)
        if load_result.missing_keys:
            raise RuntimeError(f"QAT graph mismatch: missing={len(load_result.missing_keys)}")
        if load_result.unexpected_keys:
            raise RuntimeError(f"QAT graph mismatch: unexpected={len(load_result.unexpected_keys)}")

        prepared.apply(torch.ao.quantization.disable_observer)
        self.model = prepared.to(device)
        self.reference_model = reference_model.eval()
        self.config_path = config_path

        self.task = args.task

    def infer(self, image: torch.Tensor) -> InferenceOutput:
        with torch.inference_mode():
            raw_predictions = self.model(image)
            return decode_predictions(raw_predictions, self.reference_model, self.task)


class QuantONNXBackend:
    def __init__(self, args: argparse.Namespace, reference_model, device: torch.device):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if device.type == "cuda" and "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(args.model, providers=providers)
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        input_shape = input_meta.shape
        if len(input_shape) != 4 or not all(isinstance(value, int) for value in input_shape[-2:]):
            raise RuntimeError(f"QuantONNX must have fixed NCHW spatial dimensions, got: {input_shape}")
        self.input_hw = (input_shape[-2], input_shape[-1])
        self.reference_model = reference_model.eval()
        self.device = device
        self.task = args.task
        head = self.reference_model.model[-1]
        self.box_channels = 4 * int(getattr(head, "reg_max", 1))
        self.score_channels = len(self.reference_model.names)
        self.angle_channels = int(getattr(head, "ne", 0))
        self.keypoint_channels = int(getattr(head, "nk", 0))

    def infer(self, image: torch.Tensor) -> InferenceOutput:
        outputs = self.session.run(None, {self.input_name: image.cpu().numpy()})
        output_names = [meta.name for meta in self.session.get_outputs()]
        boxes: dict[int, torch.Tensor] = {}
        scores: dict[int, torch.Tensor] = {}
        obb_boxes = obb_scores = obb_angle = None
        pose_boxes = pose_scores = pose_keypoints = None
        mask_coefficient = None
        proto_masks = None
        proto_semseg = None
        nm = int(getattr(self.reference_model.model[-1], "nm", 0))
        for name, output in zip(output_names, outputs):
            tensor = torch.from_numpy(output).to(self.device)
            if self.task == "obb" and tensor.ndim == 3 and tensor.shape[0] == 1:
                if tensor.shape[1] == self.box_channels:
                    obb_boxes = tensor
                elif tensor.shape[1] == self.score_channels:
                    obb_scores = tensor
                elif tensor.shape[1] == self.angle_channels:
                    obb_angle = tensor
            elif self.task == "pose" and tensor.ndim == 3 and tensor.shape[0] == 1:
                if tensor.shape[1] == self.box_channels:
                    pose_boxes = tensor
                elif tensor.shape[1] == self.score_channels:
                    pose_scores = tensor
                elif tensor.shape[1] == self.keypoint_channels:
                    pose_keypoints = tensor
            elif tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[1] == self.box_channels:
                boxes[tensor.shape[2]] = tensor
            elif tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[1] == self.score_channels:
                scores[tensor.shape[2]] = tensor
            elif self.task == "segment" and tensor.ndim == 3 and tensor.shape[1] == nm:
                mask_coefficient = tensor
            elif self.task == "segment" and tensor.ndim == 4:
                if name == "proto_masks" or (proto_masks is None and tensor.shape[1] == nm):
                    proto_masks = tensor
                elif name == "proto_semseg" or tensor.shape[1] == len(self.reference_model.names):
                    proto_semseg = tensor

        if self.task == "obb":
            if obb_boxes is None or obb_scores is None or obb_angle is None:
                raise RuntimeError("OBB QuantONNX must provide concatenated boxes, scores, and angle outputs")
            if obb_boxes.shape[2] != obb_scores.shape[2] or obb_boxes.shape[2] != obb_angle.shape[2]:
                raise RuntimeError("OBB QuantONNX outputs must use the same anchor count")
            head = self.reference_model.model[-1]
            strides = [int(value) for value in head.stride.tolist()]
            feats = [
                torch.zeros(1, 1, self.input_hw[0] // stride, self.input_hw[1] // stride, device=self.device)
                for stride in strides
            ]
            raw_predictions = {
                "one2one": {"boxes": obb_boxes, "scores": obb_scores, "angle": obb_angle, "feats": feats}
            }
            return decode_predictions(raw_predictions, self.reference_model, self.task)

        if self.task == "pose":
            if pose_boxes is None or pose_scores is None or pose_keypoints is None:
                raise RuntimeError("Pose QuantONNX must provide concatenated boxes, scores, and keypoints outputs")
            if pose_boxes.shape[2] != pose_scores.shape[2] or pose_boxes.shape[2] != pose_keypoints.shape[2]:
                raise RuntimeError("Pose QuantONNX outputs must use the same anchor count")
            head = self.reference_model.model[-1]
            strides = [int(value) for value in head.stride.tolist()]
            feats = [
                torch.zeros(1, 1, self.input_hw[0] // stride, self.input_hw[1] // stride, device=self.device)
                for stride in strides
            ]
            raw_predictions = {
                "one2one": {"boxes": pose_boxes, "scores": pose_scores, "kpts": pose_keypoints, "feats": feats}
            }
            return decode_predictions(raw_predictions, self.reference_model, self.task)

        if boxes.keys() != scores.keys() or len(boxes) != 3:
            raise RuntimeError("Expected three matching box/score outputs from one2one QuantONNX")
        if self.task == "segment" and (mask_coefficient is None or proto_masks is None):
            raise RuntimeError("Segment QuantONNX must provide mask_coefficient and proto_masks outputs")

        anchors = sorted(boxes, reverse=True)
        # Anchor counts are ordered largest-first (stride 8, 16, 32); derive feature-map shapes from
        # the actual input H/W so non-square inputs (e.g. 352x640) decode correctly.
        strides = [int(value) for value in self.reference_model.model[-1].stride.tolist()]
        feats = [
            torch.zeros(1, 1, self.input_hw[0] // stride, self.input_hw[1] // stride, device=self.device)
            for stride in strides
        ]
        pred_dict = {
            "boxes": [boxes[anchor_count] for anchor_count in anchors],
            "scores": [scores[anchor_count] for anchor_count in anchors],
            "feats": feats,
        }
        if self.task == "segment":
            pred_dict["mask_coefficient"] = mask_coefficient
            pred_dict["proto"] = (proto_masks, proto_semseg) if proto_semseg is not None else proto_masks
        raw_predictions = {"one2one": pred_dict} if self.reference_model.model[-1].end2end else pred_dict
        return decode_predictions(raw_predictions, self.reference_model, self.task)


def preprocess(image: np.ndarray, input_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    resized = LetterBox(new_shape=input_hw, auto=False, scaleup=True, stride=32)(image=image)
    rgb_chw = resized[..., ::-1].transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb_chw)).to(device=device, dtype=torch.float32) / 255.0
    return tensor.unsqueeze(0)


def make_result(
    inference: InferenceOutput,
    original: np.ndarray,
    image_path: Path,
    names,
    input_hw: tuple[int, int],
    confidence: float,
    max_det: int,
    task: str,
) -> Results:
    detections = inference.predictions[0]
    detections = detections[detections[:, 4] >= confidence][:max_det].clone()
    masks = None
    if len(detections):
        if task == "segment":
            if inference.proto is None:
                raise RuntimeError("Segmentation inference did not return mask prototypes")
            masks = ops.process_mask(
                inference.proto[0], detections[:, 6:], detections[:, :4], input_hw, upsample=True
            )
        detections[:, :4] = ops.scale_boxes(
            input_hw, detections[:, :4], original.shape[:2], xywh=task == "obb"
        )
    if task == "obb":
        obb = torch.cat([detections[:, :4], detections[:, 6:7], detections[:, 4:6]], dim=1)
        return Results(original, str(image_path), names, obb=obb.cpu())
    if task == "pose":
        keypoints = detections[:, 6:].reshape(len(detections), (detections.shape[1] - 6) // 3, 3)
        keypoints = ops.scale_coords(input_hw, keypoints, original.shape[:2])
        return Results(original, str(image_path), names, boxes=detections[:, :6].cpu(), keypoints=keypoints.cpu())
    return Results(
        original,
        str(image_path),
        names,
        boxes=detections[:, :6].cpu(),
        masks=masks.cpu() if masks is not None else None,
    )


def print_result(index: int, result: Results, elapsed_ms: float) -> None:
    if result.obb is not None:
        obb = result.obb
        count = len(obb)
        print(f"[{index}] {result.path}: detections={count}, inference={elapsed_ms:.2f} ms")
        if not count:
            print("  no detections")
            return
        class_ids = obb.cls.int().tolist()
        counts = Counter(result.names[class_id] for class_id in class_ids)
        print("  summary: " + ", ".join(f"{name}={number}" for name, number in sorted(counts.items())))
        for detection_index, detection in enumerate(obb.data, start=1):
            x, y, width, height, angle, confidence, class_id = detection.tolist()
            print(
                f"  #{detection_index}: class={result.names[int(class_id)]} id={int(class_id)} "
                f"conf={confidence:.4f} xywhr=({x:.1f}, {y:.1f}, {width:.1f}, {height:.1f}, {angle:.3f})"
            )
        return
    boxes = result.boxes
    count = len(boxes)
    mask_count = len(result.masks) if result.masks is not None else 0
    kpt_count = len(result.keypoints) if result.keypoints is not None else 0
    print(
        f"[{index}] {result.path}: detections={count}, masks={mask_count}, "
        f"keypoints={kpt_count}, inference={elapsed_ms:.2f} ms"
    )
    if not count:
        print("  no detections")
        return

    class_ids = boxes.cls.int().tolist()
    counts = Counter(result.names[class_id] for class_id in class_ids)
    print("  summary: " + ", ".join(f"{name}={number}" for name, number in sorted(counts.items())))
    for detection_index, box in enumerate(boxes, start=1):
        class_id = int(box.cls)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(
            f"  #{detection_index}: class={result.names[class_id]} id={class_id} conf={float(box.conf):.4f} "
            f"xyxy=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )


def _classify_names(args: argparse.Namespace):
    """Resolve class-index names for the classification head output."""
    data = args.data
    if data is None and Path(args.model).suffix.lower() == ".pt":
        checkpoint = torch.load(args.model, weights_only=False, map_location="cpu")
        data = (checkpoint.get("train_args") or {}).get("data")
    if data:
        from ultralytics.data.utils import check_cls_dataset

        return check_cls_dataset(data)["names"]
    return None


def _build_classify_qat_backend(args: argparse.Namespace, device: torch.device):
    """Return a prepared classify QAT graph that emits softmax probabilities."""
    from ultralytics.data.utils import check_cls_dataset

    checkpoint = torch.load(args.model, weights_only=False, map_location="cpu")
    train_args = checkpoint.get("train_args") or {}
    nc = len(check_cls_dataset(train_args["data"])["names"])
    yolo = YOLO(args.model_yaml, task="classify")
    yolo.model = yolo.task_map["classify"]["model"](args.model_yaml, nc=nc, ch=3)
    yolo.load(args.pretrained)
    float_model = yolo.model.float().to(device).train()
    config_path = resolve_qat_config_path(args.quant_config or train_args.get("qat_config", ""))
    if not config_path.is_file():
        raise FileNotFoundError(f"QAT config not found: {config_path}")
    _, prepared = prepare_pt2e_qat_model(
        float_model=float_model,
        device=device,
        config_path=str(config_path),
        imgsz=args.imgsz,
        dynamic_batch_max=int(train_args.get("qat_dynamic_batch_max", 128)),
    )
    prepared = BaseValidator._prepare_pt2e_model_for_eval(prepared)
    qat_state = checkpoint.get("qat_ema") or checkpoint.get("qat_model")
    if qat_state is None:
        raise KeyError("Checkpoint has no qat_ema or qat_model state")
    load_result = prepared.load_state_dict(qat_state, strict=False)
    if load_result.missing_keys:
        raise RuntimeError(f"QAT graph mismatch: missing={len(load_result.missing_keys)}")
    prepared.apply(torch.ao.quantization.disable_observer)
    prepared.to(device)

    def infer(image: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return prepared(image).softmax(1)[0].float().cpu()

    return infer, (args.imgsz, args.imgsz)


def _build_classify_onnx_backend(args: argparse.Namespace, device: torch.device):
    """Return an ORT-backed classify inference callable; QuantONNX already emits probabilities."""
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(args.model, providers=providers)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_hw = (input_meta.shape[-2], input_meta.shape[-1])
    if not all(isinstance(value, int) for value in input_hw):
        raise RuntimeError(f"Classify QuantONNX must have fixed spatial dimensions, got: {input_meta.shape}")

    def infer(image: torch.Tensor) -> torch.Tensor:
        output = session.run(None, {input_name: image.cpu().numpy()})[0]
        # Deploy output is raw logits; apply softmax on host (deploy contract) for readable top-k.
        return torch.from_numpy(output)[0].float().softmax(0)

    return infer, input_hw


def run_classify_inference(args: argparse.Namespace, device: torch.device) -> None:
    """Classify each image and save Top-5 annotated results (checkpoint or QuantONNX)."""
    from PIL import Image

    from ultralytics.data.augment import classify_transforms

    image_paths = find_images(args.source)
    names = _classify_names(args)
    if Path(args.model).suffix.lower() == ".pt":
        infer, input_hw = _build_classify_qat_backend(args, device)
        print(f"Loaded classify QAT checkpoint: {args.model}")
    else:
        infer, input_hw = _build_classify_onnx_backend(args, device)
        print(f"Loaded classify QuantONNX: {args.model} (input={input_hw[0]}x{input_hw[1]})")

    transform = classify_transforms(input_hw[0])
    output_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, mkdir=True)
    for index, image_path in enumerate(image_paths, start=1):
        original = cv2.imread(str(image_path))
        if original is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = transform(Image.fromarray(original[..., ::-1])).unsqueeze(0).to(device)
        start = time.perf_counter()
        probs = infer(image)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        display_names = names or {i: str(i) for i in range(probs.numel())}
        result = Results(original, str(image_path), display_names, probs=probs)
        topk = probs.argsort(descending=True)[:5].tolist()
        summary = ", ".join(f"{display_names[c]}={float(probs[c]):.3f}" for c in topk)
        print(f"[{index}] {image_path}: top5=[{summary}], inference={elapsed_ms:.2f} ms")
        result.save(filename=str(output_dir / image_path.name))

    print(f"Processed {len(image_paths)} image(s). Annotated results saved to: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    default_yaml, default_pretrained = TASK_DEFAULTS[args.task]
    args.model_yaml = args.model_yaml or default_yaml
    args.pretrained = args.pretrained or default_pretrained
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if model_path.suffix.lower() not in {".pt", ".onnx"}:
        raise ValueError("--model must be a QAT .pt checkpoint or task-compatible QuantONNX .onnx")

    device = select_device(args.device)
    if args.task == "classify":
        run_classify_inference(args, device)
        return

    image_paths = find_images(args.source)
    reference_model = build_reference_model(args.model_yaml, args.pretrained, device, args.task)

    if model_path.suffix.lower() == ".pt":
        backend = QATCheckpointBackend(args, reference_model, device)
        input_hw = (args.imgsz, args.imgsz)
        print(f"Loaded QAT checkpoint: {model_path} (config={backend.config_path})")
    else:
        backend = QuantONNXBackend(args, reference_model, device)
        input_hw = backend.input_hw
        print(f"Loaded QuantONNX: {model_path} (input={input_hw[0]}x{input_hw[1]})")

    output_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, mkdir=True)
    labels_dir = output_dir / "labels"
    if args.save_txt:
        labels_dir.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(image_paths, start=1):
        original = cv2.imread(str(image_path))
        if original is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = preprocess(original, input_hw, device)

        start = time.perf_counter()
        predictions = backend.infer(image)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result = make_result(
            predictions,
            original,
            image_path,
            reference_model.names,
            input_hw,
            args.conf,
            args.max_det,
            args.task,
        )
        print_result(index, result, elapsed_ms)

        result.save(filename=str(output_dir / image_path.name), line_width=args.line_width)
        if args.save_txt:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                label_path.unlink()
            result.save_txt(label_path, save_conf=args.save_conf)

    print(f"Processed {len(image_paths)} image(s). Annotated results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
