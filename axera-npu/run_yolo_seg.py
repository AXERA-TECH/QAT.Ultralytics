#!/usr/bin/env python3
"""Run a YOLO segmentation QuantONNX model and write COCO predictions."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_yolo_detect import (
    COCO80_NAMES,
    LetterBox,
    _color_for_class,
    coco80_to_coco91_class,
    create_inference_session,
    draw_detections,
    scale_boxes,
    sigmoid,
    xyxy2ltwh,
)

BOX_OUTPUTS = ("boxes_p3", "boxes_p4", "boxes_p5")
SCORE_OUTPUTS = ("scores_p3", "scores_p4", "scores_p5")
MASK_COEFFICIENT_OUTPUT = "mask_coefficient"
PROTO_MASK_OUTPUT = "proto_masks"


def find_images(folder: Path) -> list[Path]:
    image_extensions = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
    return sorted(path for path in folder.rglob("*") if path.suffix.lower() in image_extensions)


def encode_mask(mask: np.ndarray) -> dict:
    """Encode a binary HxW mask using COCO's compact Fortran-order RLE."""
    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise RuntimeError("COCO segmentation JSON requires pycocotools.") from error

    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    return {"size": [int(value) for value in encoded["size"]], "counts": counts.decode("ascii")}


def decode_segment_outputs(
    boxes_by_scale: list[np.ndarray],
    scores_by_scale: list[np.ndarray],
    mask_coefficients: np.ndarray,
    imgsz: tuple[int, int],
    max_det: int,
) -> np.ndarray:
    """Decode raw segmentation boxes, scores and mask coefficients into top-k detections."""
    if len(boxes_by_scale) != 3 or len(scores_by_scale) != 3:
        raise ValueError("Expected P3/P4/P5 box and score outputs.")
    if imgsz[0] != imgsz[1]:
        raise ValueError(f"Only square segmentation inputs are supported, got {imgsz}.")
    if mask_coefficients.ndim != 3 or mask_coefficients.shape[0] != 1:
        raise ValueError(f"Unexpected mask_coefficient shape: {mask_coefficients.shape}")

    decoded_boxes, decoded_scores = [], []
    expected_anchors = 0
    for boxes, scores in zip(boxes_by_scale, scores_by_scale):
        if boxes.ndim != 3 or boxes.shape[:2] != (1, 4):
            raise ValueError(f"Unexpected box output shape: {boxes.shape}")
        if scores.ndim != 3 or scores.shape[0] != 1:
            raise ValueError(f"Unexpected score output shape: {scores.shape}")
        if boxes.shape[2] != scores.shape[2]:
            raise ValueError(f"Box/score anchor count differs: {boxes.shape} vs {scores.shape}")

        anchor_count = boxes.shape[2]
        feature_size = int(np.sqrt(anchor_count))
        if feature_size * feature_size != anchor_count:
            raise ValueError(f"Output anchor count must be square, got {anchor_count}")
        stride = imgsz[0] // feature_size
        if stride * feature_size != imgsz[0]:
            raise ValueError(f"Input size {imgsz[0]} is incompatible with feature size {feature_size}")

        gy, gx = np.meshgrid(
            np.arange(feature_size, dtype=np.float32), np.arange(feature_size, dtype=np.float32), indexing="ij"
        )
        anchors = np.stack((gx + 0.5, gy + 0.5), axis=-1).reshape(-1, 2)
        distances = boxes[0].T
        xyxy = np.concatenate((anchors - distances[:, :2], anchors + distances[:, 2:]), axis=1) * stride
        decoded_boxes.append(xyxy)
        decoded_scores.append(sigmoid(scores[0].T))
        expected_anchors += anchor_count

    if mask_coefficients.shape[2] != expected_anchors:
        raise ValueError(
            f"mask_coefficient anchor count {mask_coefficients.shape[2]} does not match boxes {expected_anchors}"
        )

    boxes = np.concatenate(decoded_boxes, axis=0).astype(np.float32, copy=False)
    scores = np.concatenate(decoded_scores, axis=0).astype(np.float32, copy=False)
    coefficients = mask_coefficients[0].T.astype(np.float32, copy=False)
    if coefficients.shape[0] != boxes.shape[0]:
        raise ValueError("Mask coefficients are not aligned with decoded boxes.")

    topk = min(max_det, boxes.shape[0])
    anchor_index = np.argsort(-scores.max(axis=1), kind="stable")[:topk]
    boxes, scores, coefficients = boxes[anchor_index], scores[anchor_index], coefficients[anchor_index]
    class_count = scores.shape[1]
    score_index = np.argsort(-scores.reshape(-1), kind="stable")[:topk]
    selected = score_index // class_count
    classes = (score_index % class_count).astype(np.float32)
    return np.concatenate(
        (boxes[selected], scores.reshape(-1)[score_index, None], classes[:, None], coefficients[selected]), axis=1
    )


def process_masks(
    prototypes: np.ndarray, coefficients: np.ndarray, boxes: np.ndarray, input_hw: tuple[int, int]
) -> np.ndarray:
    """Apply mask coefficients, crop in prototype space, and upsample to the model input with NumPy/OpenCV."""
    channels, mask_h, mask_w = prototypes.shape
    if coefficients.shape[1] != channels:
        raise ValueError(f"Mask coefficient channels {coefficients.shape[1]} do not match prototypes {channels}")
    masks = (coefficients @ prototypes.reshape(channels, -1)).reshape(-1, mask_h, mask_w)
    ratios = np.array([mask_w / input_hw[1], mask_h / input_hw[0]] * 2, dtype=np.float32)
    scaled_boxes = np.rint(boxes * ratios).astype(np.int32)
    for mask, (x1, y1, x2, y2) in zip(masks, scaled_boxes):
        x1, x2 = np.clip((x1, x2), 0, mask_w)
        y1, y2 = np.clip((y1, y2), 0, mask_h)
        mask[:y1] = 0
        mask[y2:] = 0
        mask[:, :x1] = 0
        mask[:, x2:] = 0
    upsampled = np.stack(
        [cv2.resize(mask, (input_hw[1], input_hw[0]), interpolation=cv2.INTER_LINEAR) for mask in masks]
    )
    return (upsampled > 0.0).astype(np.uint8)


def scale_masks(masks: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """Undo letterbox padding and resize binary input masks to the original image with NumPy/OpenCV."""
    input_h, input_w = masks.shape[1:]
    original_h, original_w = original_shape
    gain = min(input_h / original_h, input_w / original_w)
    pad_w = (input_w - original_w * gain) / 2
    pad_h = (input_h - original_h * gain) / 2
    top, left = round(pad_h - 0.1), round(pad_w - 0.1)
    bottom, right = input_h - round(pad_h + 0.1), input_w - round(pad_w + 0.1)
    cropped = masks[:, top:bottom, left:right].astype(np.float32)
    resized = np.stack([cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_LINEAR) for mask in cropped])
    return (resized > 0.5).astype(np.uint8)


def draw_segmentations(
    image: np.ndarray, masks: np.ndarray, scores: np.ndarray, classes: np.ndarray, conf: float
) -> np.ndarray:
    """Overlay masks first, then reuse the detection renderer for boxes and labels."""
    visualized = image.copy()
    for mask, score, class_id in zip(masks, scores, classes):
        if float(score) < conf:
            continue
        color = np.asarray(_color_for_class(int(class_id)), dtype=np.float32)
        visualized[mask.astype(bool)] = (0.5 * visualized[mask.astype(bool)] + 0.5 * color).astype(np.uint8)
    return visualized


class YOLOSegPredictor:
    """Standalone COCO segmentation predictor for export.py's nine-output model contract."""

    def __init__(self, model_path: str, conf_thres: float, max_det: int, runtime: str = "auto"):
        self.session, self.runtime = create_inference_session(model_path, runtime)
        input_meta = self.session.get_inputs()[0]
        if len(input_meta.shape) != 4 or not all(isinstance(value, int) for value in input_meta.shape[-2:]):
            raise ValueError(f"Expected fixed NCHW model input, got {input_meta.shape}")
        self.input_name = input_meta.name
        self.input_hw = (input_meta.shape[-2], input_meta.shape[-1])
        self.output_names = [meta.name for meta in self.session.get_outputs()]
        required = {*BOX_OUTPUTS, *SCORE_OUTPUTS, MASK_COEFFICIENT_OUTPUT, PROTO_MASK_OUTPUT}
        missing = required.difference(self.output_names)
        if missing:
            raise ValueError(f"Model does not match export.py segmentation outputs; missing: {sorted(missing)}")

        self.letterbox = LetterBox(new_shape=self.input_hw, auto=False, stride=32)
        self.conf_thres = conf_thres
        self.max_det = max_det
        self.class_map = coco80_to_coco91_class()
        self.predictions: list[dict] = []
        self.infer_ms = 0.0
        self.save_vis: Path | None = None
        self.vis_conf = 0.25
        self.vis_limit = 0
        self.vis_count = 0

    def preprocess(self, image_path: Path) -> tuple[np.ndarray, dict]:
        original = cv2.imread(str(image_path))
        if original is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        image = self.letterbox(original)[..., ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image, dtype=np.float32) / 255.0
        return image[None], {"image_name": image_path.stem, "original": original, "orig_shape": original.shape[:2]}

    def postprocess(self, outputs: list[np.ndarray], meta: dict) -> None:
        named = dict(zip(self.output_names, outputs))
        prediction = decode_segment_outputs(
            [named[name] for name in BOX_OUTPUTS],
            [named[name] for name in SCORE_OUTPUTS],
            named[MASK_COEFFICIENT_OUTPUT],
            self.input_hw,
            self.max_det,
        )
        prediction = prediction[prediction[:, 4] > self.conf_thres]
        if not len(prediction):
            return

        masks = process_masks(named[PROTO_MASK_OUTPUT][0], prediction[:, 6:], prediction[:, :4], self.input_hw)
        masks = scale_masks(masks, meta["orig_shape"])
        prediction[:, :4] = scale_boxes(self.input_hw, prediction[:, :4], meta["orig_shape"])
        boxes = xyxy2ltwh(prediction[:, :4])
        scores = prediction[:, 4]
        classes = prediction[:, 5].astype(np.int32)

        if self.save_vis is not None and self.vis_count < self.vis_limit:
            visualized = draw_segmentations(meta["original"], masks, scores, classes, self.vis_conf)
            visualized = draw_detections(visualized, prediction[:, :4], scores, classes, COCO80_NAMES, self.vis_conf)
            cv2.imwrite(str(self.save_vis / f"{meta['image_name']}.jpg"), visualized)
            self.vis_count += 1

        image_id = int(meta["image_name"]) if meta["image_name"].isnumeric() else meta["image_name"]
        for box, score, class_id, mask in zip(boxes, scores, classes, masks):
            self.predictions.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(class_id)],
                    "bbox": [round(float(value), 3) for value in box],
                    "score": round(float(score), 5),
                    "segmentation": encode_mask(mask),
                }
            )

    def run_image(self, image_path: Path) -> None:
        image, meta = self.preprocess(image_path)
        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: image})
        self.infer_ms += (time.perf_counter() - start) * 1000.0
        self.postprocess(outputs, meta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone YOLO segmentation model evaluator.")
    parser.add_argument("--model", required=True, help="Nine-output segmentation model exported by export.py.")
    parser.add_argument("--img-dir", required=True, help="Image directory, such as COCO val2017.")
    parser.add_argument("--output-json", default="output/onnx_preds-yolo26n-seg.json", help="COCO result JSON path.")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument(
        "--runtime",
        choices=("auto", "onnxruntime", "axengine"),
        default="auto",
        help="auto prefers axengine when installed; onnxruntime uses CUDA/CPU providers.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N images; 0 runs all images.")
    parser.add_argument("--save-vis", default="", help="Directory for visualized masks and boxes; empty disables it.")
    parser.add_argument("--vis-conf", type=float, default=0.25, help="Confidence threshold for visualizations.")
    parser.add_argument("--vis-limit", type=int, default=50, help="Maximum visualized images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.img_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")
    if args.max_det < 1:
        raise ValueError("--max-det must be positive")

    predictor = YOLOSegPredictor(args.model, args.conf_thres, args.max_det, args.runtime)
    print(f"runtime: {predictor.runtime}")
    if args.save_vis:
        predictor.save_vis = Path(args.save_vis)
        predictor.save_vis.mkdir(parents=True, exist_ok=True)
        predictor.vis_conf = args.vis_conf
        predictor.vis_limit = args.vis_limit

    images = find_images(image_dir)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"No supported images found in: {image_dir}")
    for image_path in tqdm.tqdm(images, desc="infer"):
        predictor.run_image(image_path)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictor.predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"images: {len(images)}")
    print(f"predictions: {len(predictor.predictions)}")
    print(f"avg_infer_ms: {predictor.infer_ms / len(images):.3f}")
    print(f"saved json to: {output_path}")


if __name__ == "__main__":
    main()
