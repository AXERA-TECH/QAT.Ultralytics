#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tqdm

COCO80_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def create_inference_session(model_path: str, runtime: str):
    """Create an ONNX Runtime or AxEngine session without mixing provider names."""
    if runtime not in {"auto", "onnxruntime", "axengine"}:
        raise ValueError(f"Unsupported runtime: {runtime}")

    axengine_error = None
    if runtime in {"auto", "axengine"}:
        try:
            import axengine as axengine_ort

            return axengine_ort.InferenceSession(model_path, providers=["AxEngineExecutionProvider"]), "axengine"
        except ImportError as error:
            axengine_error = error
        except Exception:
            if runtime == "axengine":
                raise

    if runtime == "axengine":
        raise RuntimeError(
            "axengine is not installed; install the Axera runtime or use --runtime onnxruntime."
        ) from axengine_error

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is not installed; install it or use --runtime axengine.") from error
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in ort.get_available_providers()
    ]
    return ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"]), "onnxruntime"


def _color_for_class(cls_id: int) -> tuple:
    return (int(41 * cls_id + 60) % 256, int(97 * cls_id + 120) % 256, int(53 * cls_id + 30) % 256)


def draw_detections(image, boxes_xyxy, scores, classes, class_names, conf_thres=0.25):
    vis = image.copy()
    order = np.argsort(np.asarray(scores))
    for i in order:
        score = float(scores[i])
        if score < conf_thres:
            continue
        cls_id = int(classes[i])
        x1, y1, x2, y2 = (round(float(v)) for v in boxes_xyxy[i])
        color = _color_for_class(cls_id)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        label = f"{name} {score:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_top = max(y1, th + bl + 2)
        cv2.rectangle(vis, (x1, y_top - th - bl - 2), (x1 + tw + 2, y_top), color, -1)
        cv2.putText(
            vis, label, (x1 + 1, y_top - bl - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
    return vis


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


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def xyxy2ltwh(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def clip_boxes(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape[:2]
    boxes[..., 0] = np.clip(boxes[..., 0], 0, w)
    boxes[..., 1] = np.clip(boxes[..., 1], 0, h)
    boxes[..., 2] = np.clip(boxes[..., 2], 0, w)
    boxes[..., 3] = np.clip(boxes[..., 3], 0, h)
    return boxes


def scale_boxes(img1_shape: tuple[int, int], boxes: np.ndarray, img0_shape: tuple[int, int]) -> np.ndarray:
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
    pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    boxes[..., 0] -= pad_x
    boxes[..., 1] -= pad_y
    boxes[..., 2] -= pad_x
    boxes[..., 3] -= pad_y
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)


def split_detection_outputs(outputs: list[np.ndarray], num_classes: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pair P3/P4/P5 regression and classification tensors by anchor count, not ONNX output name/order.

    C++ ports should apply the same rule: classification tensors use C=num_classes; regression tensors use C=4 (YOLO26)
    or C=4*reg_max (YOLO11). The matching N dimension identifies the corresponding feature scale.
    """
    regression, classification = {}, {}
    for output in outputs:
        tensor = np.asarray(output)
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            continue
        channels, anchors = tensor.shape[1:]
        if channels == num_classes:
            if anchors in classification:
                raise ValueError(f"Duplicate classification output for {anchors} anchors")
            classification[anchors] = tensor
        elif channels >= 4 and channels % 4 == 0:
            if anchors in regression:
                raise ValueError(f"Duplicate regression output for {anchors} anchors")
            regression[anchors] = tensor

    if len(regression) != 3 or set(regression) != set(classification):
        raise ValueError(
            "Expected three matching detection regression/classification outputs in BCN layout; "
            f"regression={sorted(regression)}, classification={sorted(classification)}"
        )
    return [(regression[anchors], classification[anchors]) for anchors in sorted(regression, reverse=True)]


def decode_yolo26_distances(regression: np.ndarray) -> np.ndarray:
    """Convert the current YOLO26 [1, 4, N] ltrb-distance output to [N, 4]."""
    if regression.ndim != 3 or regression.shape[0] != 1:
        raise ValueError(f"Expected BCN regression tensor, got {regression.shape}")
    if regression.shape[1] != 4:
        raise ValueError(f"YOLO26 decoded distances require C=4, got {regression.shape}")
    return regression[0].T.astype(np.float32, copy=False)


def decode_yolo11_dfl(regression: np.ndarray) -> np.ndarray:
    """Decode YOLO11 [1, 4*reg_max, N] DFL logits to [N, left, top, right, bottom].

    For every anchor and side, the C++ equivalent is: ``distance = sum(softmax(logits[0:reg_max]) * bin_index)``.
    """
    if regression.ndim != 3 or regression.shape[0] != 1:
        raise ValueError(f"Expected BCN regression tensor, got {regression.shape}")
    channels, anchor_count = regression.shape[1:]
    if channels <= 4 or channels % 4:
        raise ValueError(f"YOLO11 DFL logits require C=4*reg_max and reg_max>1, got {regression.shape}")

    reg_max = channels // 4
    logits = regression[0].reshape(4, reg_max, anchor_count).transpose(2, 0, 1).astype(np.float32, copy=True)
    logits -= logits.max(axis=2, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    return (probabilities * np.arange(reg_max, dtype=np.float32)).sum(axis=2)


def decode_regression(regression: np.ndarray) -> np.ndarray:
    """Dispatch current YOLO26 decoded distances and YOLO11 DFL logits by regression channel count."""
    if regression.ndim != 3 or regression.shape[0] != 1:
        raise ValueError(f"Expected BCN regression tensor, got {regression.shape}")
    return decode_yolo26_distances(regression) if regression.shape[1] == 4 else decode_yolo11_dfl(regression)


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Return IoU between one xyxy box and N xyxy boxes."""
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.clip(bottom_right - top_left, 0.0, None).prod(axis=1)
    area = np.clip(box[2] - box[0], 0.0, None) * np.clip(box[3] - box[1], 0.0, None)
    areas = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    return intersection / np.maximum(area + areas - intersection, np.finfo(np.float32).eps)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    max_nms: int = 30000,
) -> np.ndarray:
    """Apply multi-label, class-aware NMS to conventional YOLO BCN head outputs."""
    anchor_index, class_index = np.nonzero(scores > conf_thres)
    if not len(anchor_index):
        return np.empty((0, 6), dtype=np.float32)

    candidate_scores = scores[anchor_index, class_index]
    if len(candidate_scores) > max_nms:
        order = np.argsort(-candidate_scores, kind="stable")[:max_nms]
        anchor_index, class_index, candidate_scores = (
            anchor_index[order],
            class_index[order],
            candidate_scores[order],
        )
    candidate_boxes = boxes[anchor_index]

    kept = []
    for cls_id in np.unique(class_index):
        indices = np.flatnonzero(class_index == cls_id)
        order = indices[np.argsort(-candidate_scores[indices], kind="stable")]
        while order.size:
            selected = order[0]
            kept.append(selected)
            order = order[1:]
            if order.size:
                order = order[box_iou_one_to_many(candidate_boxes[selected], candidate_boxes[order]) <= iou_thres]

    kept = np.asarray(kept, dtype=np.int64)
    kept = kept[np.argsort(-candidate_scores[kept], kind="stable")[:max_det]]
    return np.concatenate(
        (
            candidate_boxes[kept],
            candidate_scores[kept, None],
            class_index[kept, None].astype(np.float32),
        ),
        axis=1,
    ).astype(np.float32, copy=False)


def decode_yolo_detection(
    outputs: list[np.ndarray],
    num_classes: int,
    max_det: int,
    input_hw: tuple[int, int],
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    head_type: str = "auto",
    output_names: list[str] | None = None,
) -> np.ndarray:
    """Decode six raw YOLO26/YOLO11 outputs into xyxy, score and class predictions.

    YOLO26 one-to-one uses head top-k selection, while YOLO26 one-to-many and YOLO11 use class-aware NMS. In auto mode,
    DFL regression and the stable ``boxes_p*``/``scores_p*`` output names select one-to-many.
    """
    if input_hw[0] != input_hw[1]:
        raise ValueError(f"Only square detection inputs are supported, got {input_hw}")
    if head_type not in {"auto", "one2one", "one2many"}:
        raise ValueError(f"Unsupported head type: {head_type}")

    predictions = []
    output_pairs = split_detection_outputs(outputs, num_classes)
    uses_dfl = any(regression.shape[1] > 4 for regression, _ in output_pairs)
    for regression, classification in output_pairs:
        anchor_count = regression.shape[2]
        feature_size = int(np.sqrt(anchor_count))
        if feature_size * feature_size != anchor_count or input_hw[0] % feature_size:
            raise ValueError(f"Cannot derive a square feature map from {anchor_count} anchors")
        stride = input_hw[0] // feature_size
        gy, gx = np.meshgrid(
            np.arange(feature_size, dtype=np.float32), np.arange(feature_size, dtype=np.float32), indexing="ij"
        )
        anchors = np.stack((gx + 0.5, gy + 0.5), axis=-1).reshape(-1, 2)
        # Both branches return ltrb distances in feature-grid units. Convert them to input-image xyxy coordinates.
        distances = decode_regression(regression)
        xyxy = np.concatenate((anchors - distances[:, :2], anchors + distances[:, 2:]), axis=1) * stride
        predictions.append(np.concatenate((xyxy, sigmoid(classification[0].T)), axis=1))

    prediction = np.concatenate(predictions, axis=0).astype(np.float32, copy=False)
    boxes, scores = prediction[:, :4], prediction[:, 4:]
    names = output_names or []
    has_named_one2many_outputs = any(name.startswith("boxes_p") for name in names) and any(
        name.startswith("scores_p") for name in names
    )
    use_nms = head_type == "one2many" or (head_type == "auto" and (uses_dfl or has_named_one2many_outputs))
    if use_nms:
        return class_aware_nms(boxes, scores, conf_thres, iou_thres, max_det)

    topk = min(max_det, boxes.shape[0])
    anchor_index = np.argsort(-scores.max(axis=1), kind="stable")[:topk]
    boxes, scores = boxes[anchor_index], scores[anchor_index]
    score_index = np.argsort(-scores.reshape(-1), kind="stable")[:topk]
    classes = (score_index % num_classes).astype(np.float32)
    return np.concatenate(
        (boxes[score_index // num_classes], scores.reshape(-1)[score_index, None], classes[:, None]), axis=1
    )


class YOLODetectPredictor:
    def __init__(
        self,
        model_path: str,
        conf_thres: float = 0.001,
        iou_thres: float = 0.7,
        max_det: int = 300,
        runtime: str = "auto",
        head_type: str = "auto",
    ):
        if head_type not in {"auto", "one2one", "one2many"}:
            raise ValueError(f"Unsupported head type: {head_type}")
        self.model_path = model_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.head_type = head_type
        self.cls_map = coco80_to_coco91_class()
        self.session, self.runtime = create_inference_session(model_path, runtime)
        input_meta = self.session.get_inputs()[0]
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.input_name = input_meta.name
        _, _, self.input_h, self.input_w = input_meta.shape
        self.letterbox = LetterBox(new_shape=(self.input_h, self.input_w), auto=False, stride=32)
        self.infer_ms = 0.0
        self.predictions: list[dict] = []
        self.save_vis = ""
        self.vis_conf = 0.25
        self.vis_limit = 0
        self.vis_count = 0
        self.class_names = COCO80_NAMES

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
            "orig_image": image,
        }
        return img[None, ...], meta

    def postprocess(self, outputs: list[np.ndarray], meta: dict) -> None:
        pred = decode_yolo_detection(
            outputs,
            num_classes=80,
            max_det=self.max_det,
            input_hw=(self.input_h, self.input_w),
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            head_type=self.head_type,
            output_names=self.output_names,
        )
        pred = pred[pred[:, 4] > self.conf_thres]
        if pred.shape[0] == 0:
            return

        pred[:, :4] = scale_boxes((self.input_h, self.input_w), pred[:, :4], meta["orig_shape"])
        boxes = xyxy2ltwh(pred[:, :4])
        scores = pred[:, 4]
        classes = pred[:, 5].astype(np.int32)
        if self.save_vis and self.vis_count < self.vis_limit:
            vis_xyxy = pred[:, :4]
            vis_img = draw_detections(meta["orig_image"], vis_xyxy, scores, classes, self.class_names, self.vis_conf)
            cv2.imwrite(str(Path(self.save_vis) / f"{meta['image_name']}.jpg"), vis_img)
            self.vis_count += 1
        image_id = int(meta["image_name"]) if meta["image_name"].isnumeric() else meta["image_name"]

        for box, score, cls_id in zip(boxes, scores, classes):
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
    parser = argparse.ArgumentParser(description="Standalone evaluator for YOLO26/YOLO11 COCO detection models.")
    parser.add_argument("--model", required=True, help="Input detection model for the selected runtime.")
    parser.add_argument("--img-dir", required=True, help="COCO val2017 image directory.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/onnx_preds_yolo_detect.json",
        help="Output COCO prediction json.",
    )
    parser.add_argument("--conf-thres", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--iou-thres", type=float, default=0.7, help="Class-aware NMS IoU threshold for DFL outputs.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument(
        "--head-type",
        choices=("auto", "one2one", "one2many"),
        default="auto",
        help="auto uses DFL or boxes_p*/scores_p* names for one-to-many; set explicitly when an AXModel drops names.",
    )
    parser.add_argument(
        "--runtime",
        choices=("auto", "onnxruntime", "axengine"),
        default="auto",
        help="auto prefers axengine when installed; onnxruntime uses CUDA/CPU providers.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N images, 0 means all.")
    parser.add_argument(
        "--save-vis", type=str, default="", help="Directory to save visualized detections; empty disables."
    )
    parser.add_argument("--vis-conf", type=float, default=0.25, help="Confidence threshold for drawn boxes.")
    parser.add_argument("--vis-limit", type=int, default=50, help="Max number of images to visualize.")
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

    predictor = YOLODetectPredictor(
        args.model, args.conf_thres, args.iou_thres, args.max_det, args.runtime, args.head_type
    )
    print(f"runtime: {predictor.runtime}")
    if args.save_vis:
        predictor.save_vis = args.save_vis
        predictor.vis_conf = args.vis_conf
        predictor.vis_limit = args.vis_limit
        Path(args.save_vis).mkdir(parents=True, exist_ok=True)
    images = find_images(img_dir)
    if args.limit > 0:
        images = images[: args.limit]

    for image_path in tqdm.tqdm(images, desc="infer"):
        predictor.run_image(str(image_path))
    #     print(image_path)
    #     break
    # for i in predictor.predictions[:5]:
    #     print(i)
    # exit(0)
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
