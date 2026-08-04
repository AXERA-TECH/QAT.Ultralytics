#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import tqdm


def coco80_to_coco91_class() -> list[int]:
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]


class LetterBox:
    def __init__(
        self,
        new_shape: tuple[int, int] = (640, 640),
        auto: bool = False,
        scaleup: bool = True,
        center: bool = True,
        stride: int = 32,
        padding_value: int = 114,
        interpolation: int = cv2.INTER_LINEAR,
    ):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleup = scaleup
        self.center = center
        self.stride = stride
        self.padding_value = padding_value
        self.interpolation = interpolation

    def __call__(self, image: np.ndarray) -> np.ndarray:
        shape = image.shape[:2]
        r = min(self.new_shape[0] / shape[0], self.new_shape[1] / shape[1])
        if not self.scaleup:
            r = min(r, 1.0)

        new_unpad = (round(shape[1] * r), round(shape[0] * r))
        dw = self.new_shape[1] - new_unpad[0]
        dh = self.new_shape[0] - new_unpad[1]
        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)

        if self.center:
            dw /= 2
            dh /= 2

        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=self.interpolation)

        top = round(dh - 0.1) if self.center else 0
        bottom = round(dh + 0.1)
        left = round(dw - 0.1) if self.center else 0
        right = round(dw + 0.1)
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.padding_value,) * 3)


def ensure_hwc(output: np.ndarray, channels: int) -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected output rank: {output.shape}")
    if arr.shape[-1] == channels:
        return arr
    if arr.shape[0] == channels:
        return np.transpose(arr, (1, 2, 0))
    raise ValueError(f"Unexpected output layout: {output.shape}, expected channel size {channels}")


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def xyxy2ltwh(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def clip_boxes(boxes: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    h, w = shape[:2]
    boxes[..., 0].clamp_(0, w)
    boxes[..., 1].clamp_(0, h)
    boxes[..., 2].clamp_(0, w)
    boxes[..., 3].clamp_(0, h)
    return boxes


def scale_boxes(img1_shape: tuple[int, int], boxes: torch.Tensor, img0_shape: tuple[int, int]) -> torch.Tensor:
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
    pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    boxes[..., 0] -= pad_x
    boxes[..., 1] -= pad_y
    boxes[..., 2] -= pad_x
    boxes[..., 3] -= pad_y
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def batched_nms(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_thres: float, max_det: int
) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    keep = []
    for cls_id in np.unique(class_ids):
        idxs = np.where(class_ids == cls_id)[0]
        cls_boxes = boxes[idxs]
        cls_scores = scores[idxs]

        order = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[order]
        n = len(order)
        suppressed = np.zeros(n, dtype=bool)

        for i in range(n):
            if suppressed[i]:
                continue
            keep.append(idxs[order[i]])

            x1_i, y1_i, x2_i, y2_i = cls_boxes[i]
            area_i = (x2_i - x1_i) * (y2_i - y1_i)

            for j in range(i + 1, n):
                if suppressed[j]:
                    continue
                x1_j, y1_j, x2_j, y2_j = cls_boxes[j]
                inter_x1 = max(x1_i, x1_j)
                inter_y1 = max(y1_i, y1_j)
                inter_x2 = min(x2_i, x2_j)
                inter_y2 = min(y2_i, y2_j)
                if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                    continue
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area_j = (x2_j - x1_j) * (y2_j - y1_j)
                iou = inter_area / (area_i + area_j - inter_area + 1e-16)
                if iou > iou_thres:
                    suppressed[j] = True

    if not keep:
        return np.array([], dtype=np.int64)

    keep = np.array(keep)
    order = np.argsort(-scores[keep])
    return keep[order][:max_det]


def pair_outputs(outputs: list[np.ndarray], num_classes: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    groups: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for output in outputs:
        squeezed = output[0] if output.ndim == 4 and output.shape[0] == 1 else output
        if squeezed.ndim != 3:
            raise ValueError(f"Unsupported output shape: {output.shape}")

        shape = squeezed.shape
        if shape[-1] in (4, num_classes):
            spatial = (shape[0], shape[1])
            channels = shape[-1]
        elif shape[0] in (4, num_classes):
            spatial = (shape[1], shape[2])
            channels = shape[0]
        else:
            raise ValueError(f"Cannot classify output tensor shape: {output.shape}")

        group = groups.setdefault(spatial, {})
        if channels == 4:
            group["box"] = ensure_hwc(output, 4)
        else:
            group["cls"] = ensure_hwc(output, num_classes)

    pairs = []
    for (feat_h, _), group in sorted(groups.items(), reverse=True):
        if "box" not in group or "cls" not in group:
            raise ValueError("Incomplete output pair.")
        stride = 640 // feat_h
        pairs.append((stride, group["box"], group["cls"]))
    return pairs


def decode_predictions(outputs: list[np.ndarray], num_classes: int = 80) -> torch.Tensor:
    if len(outputs) == 1 and outputs[0].shape == (1, 84, 8400):
        pred = torch.from_numpy(outputs[0].astype(np.float32))[0]
        pred = pred.T
        boxes_xywh = pred[:, :4].clone()
        wh_half = boxes_xywh[:, 2:4] / 2
        xy = boxes_xywh[:, :2]
        pred[:, :4] = torch.cat([xy - wh_half, xy + wh_half], dim=1)
        return pred.unsqueeze(0)

    perscale_configs: dict[int, dict[str, np.ndarray]] = {}
    for t in outputs:
        s = t.shape
        if s[0] == 1 and s[2] in {6400, 1600, 400} and s[1] in (4, num_classes):
            n = s[2]
            entry = perscale_configs.setdefault(n, {})
            if s[1] == 4:
                entry["box"] = t
            elif s[1] == num_classes:
                entry["cls"] = t

    if len(perscale_configs) == 3:
        strides_map = {6400: 8, 1600: 16, 400: 32}
        feat_map = {6400: 80, 1600: 40, 400: 20}
        preds = []
        for n in sorted(perscale_configs, reverse=True):
            box = perscale_configs[n]["box"]
            cl = perscale_configs[n]["cls"]
            stride = strides_map[n]
            fs = feat_map[n]
            dist = box[0].T
            cl = cl[0].T
            gy, gx = np.meshgrid(np.arange(fs, dtype=np.float32), np.arange(fs, dtype=np.float32), indexing="ij")
            anchor_points = np.stack((gx + 0.5, gy + 0.5), axis=-1).reshape(-1, 2)
            x1y1 = (anchor_points - dist[:, :2]) * stride
            x2y2 = (anchor_points + dist[:, 2:]) * stride
            xyxy = np.concatenate((x1y1, x2y2), axis=1)
            cl = sigmoid(cl)
            preds.append(np.concatenate((xyxy, cl), axis=1))
        return torch.from_numpy(np.concatenate(preds, axis=0).astype(np.float32))[None, ...]

    pairs = pair_outputs(outputs, num_classes)
    preds = []
    for stride, feat_box, feat_cls in pairs:
        feat_h, feat_w, _ = feat_box.shape
        gy, gx = np.meshgrid(np.arange(feat_h, dtype=np.float32), np.arange(feat_w, dtype=np.float32), indexing="ij")
        anchor_points = np.stack((gx + 0.5, gy + 0.5), axis=-1).reshape(-1, 2)
        dist = feat_box.reshape(-1, 4)
        x1y1 = (anchor_points - dist[:, :2]) * stride
        x2y2 = (anchor_points + dist[:, 2:]) * stride
        xyxy = np.concatenate((x1y1, x2y2), axis=1)
        cl = sigmoid(feat_cls.reshape(-1, num_classes))
        preds.append(np.concatenate((xyxy, cl), axis=1))
    return torch.from_numpy(np.concatenate(preds, axis=0).astype(np.float32))[None, ...]


class YOLO26ONNXPredictor:
    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.001,
        iou_thres: float = 0.7,
        max_det: int = 300,
        max_nms: int = 30000,
    ):
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.max_nms = max_nms
        self.cls_map = coco80_to_coco91_class()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        _, _, self.input_h, self.input_w = input_meta.shape
        self.letterbox = LetterBox(new_shape=(self.input_h, self.input_w), auto=False, stride=32)
        self.infer_ms = 0.0
        self.predictions: list[dict] = []

    def preprocess(self, image_path: str) -> tuple[np.ndarray, dict]:
        image = cv2.imread(image_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        img = self.letterbox(image)
        img = img[..., ::-1].transpose((2, 0, 1))
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        meta = {
            "image_name": Path(image_path).stem,
            "orig_shape": image.shape[:2],
        }
        return img[None, ...], meta

    def postprocess(self, outputs: list[np.ndarray], meta: dict) -> None:
        pred = decode_predictions(outputs, num_classes=80)[0]
        boxes = pred[:, :4]
        scores = pred[:, 4:]

        box_indices, cls_ids = torch.where(scores > self.conf_thres)
        if box_indices.numel() == 0:
            return
        sel_boxes = boxes[box_indices].cpu().numpy()
        sel_scores = scores[box_indices, cls_ids].cpu().numpy()
        sel_cls = cls_ids.cpu().numpy().astype(np.int32)

        n = len(sel_boxes)
        if n > self.max_nms:
            topk = np.argsort(-sel_scores)[: self.max_nms]
            sel_boxes = sel_boxes[topk]
            sel_scores = sel_scores[topk]
            sel_cls = sel_cls[topk]

        keep = batched_nms(sel_boxes, sel_scores, sel_cls, self.iou_thres, self.max_det)
        if len(keep) == 0:
            return

        boxes_nms = torch.from_numpy(sel_boxes[keep])
        scores_nms = torch.from_numpy(sel_scores[keep])
        cls_nms = torch.from_numpy(sel_cls[keep].astype(np.int32))

        boxes_nms = scale_boxes((self.input_h, self.input_w), boxes_nms, meta["orig_shape"])
        boxes_np = xyxy2ltwh(boxes_nms).cpu().numpy()
        scores_np = scores_nms.cpu().numpy()
        classes_np = cls_nms.cpu().numpy().astype(np.int32)
        image_id = int(meta["image_name"])

        for box, score, cls_id in zip(boxes_np, scores_np, classes_np):
            self.predictions.append(
                {
                    "image_id": image_id,
                    "category_id": self.cls_map[int(cls_id)],
                    "bbox": [round(float(x), 3) for x in box],
                    "score": round(float(score), 5),
                }
            )

    def run_image(self, image_path: str) -> None:
        img, meta = self.preprocess(image_path)
        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: img})
        self.infer_ms += (time.perf_counter() - start) * 1000.0
        self.postprocess(outputs, meta)


def parse_args():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Standalone ONNX evaluator for YOLO26 one2many models (with NMS).")
    parser.add_argument(
        "--model",
        type=str,
        default=str(root / "onnx/yolo26n-backbone.onnx"),
        help="Input ONNX model.",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="/data/pengyancao/data/coco/val2017",
        help="COCO val2017 image directory.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(root / "output/onnx_preds-yolo26n-one2many.json"),
        help="Output COCO prediction json.",
    )
    parser.add_argument("--conf-thres", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.7, help="IoU threshold for NMS.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N images, 0 means all.")
    return parser.parse_args()


def find_images(folder: Path) -> list[Path]:
    images = sorted(folder.glob("*.jpg"))
    if not images:
        images = sorted(folder.glob("*.png"))
    return images


def main():
    args = parse_args()
    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {img_dir}")

    predictor = YOLO26ONNXPredictor(args.model, args.conf_thres, args.iou_thres, args.max_det)
    images = find_images(img_dir)
    if args.limit > 0:
        images = images[: args.limit]

    for image_path in tqdm.tqdm(images, desc="infer"):
        predictor.run_image(str(image_path))
        print(image_path)
        break
    for i in predictor.predictions[:]:
        print(i)
    sys.exit(0)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(predictor.predictions, f, ensure_ascii=False, indent=2)

    avg_ms = predictor.infer_ms / len(images) if images else 0.0
    print(f"images: {len(images)}")
    print(f"predictions: {len(predictor.predictions)}")
    print(f"avg_infer_ms: {avg_ms:.3f}")
    print(f"saved json to: {output_path}")


if __name__ == "__main__":
    main()
