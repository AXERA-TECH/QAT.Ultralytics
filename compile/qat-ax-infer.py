# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import axengine as ort_ax
import torch
from megfile import smart_exists, smart_glob
import json
import tqdm
import time
import sys
import os


def coco80_to_coco91_class():
    r"""https://github.com/ultralytics/ultralytics/blob/e3a987c2a74c4ab55e0fbef3b0d8f993cc91c198/ultralytics/data/converter.py#L125"""
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


def xyxy2xywh(x):
    """https://github.com/ultralytics/ultralytics/blob/ae859fbd8f38928056c6ab2c0f7fdb2961ea321e/ultralytics/utils/ops.py#L204"""
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2  # x center
    y[..., 1] = (y1 + y2) / 2  # y center
    y[..., 2] = x2 - x1  # width
    y[..., 3] = y2 - y1  # height
    return y


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L57"""
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def batch_probiou(obb1: torch.Tensor | np.ndarray, obb2: torch.Tensor | np.ndarray, eps: float = 1e-7) -> torch.Tensor:
    """https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L254"""
    obb1 = torch.from_numpy(obb1) if isinstance(obb1, np.ndarray) else obb1
    obb2 = torch.from_numpy(obb2) if isinstance(obb2, np.ndarray) else obb2

    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in obb2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in _get_covariance_matrix(obb2))

    t1 = (
        ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)
    ) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps)) * 0.5
    t3 = (
        ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2))
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    bd = (t1 + t2 + t3).clamp(eps, 100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    return 1 - hd


def _get_covariance_matrix(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L190"""
    # Gaussian bounding boxes, ignore the center points (the first two columns) because they are not needed here.
    gbbs = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, c = gbbs.split(1, dim=-1)
    cos = c.cos()
    sin = c.sin()
    cos2 = cos.pow(2)
    sin2 = sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def xywh2xyxy(x):
    """https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L224"""
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def empty_like(x):
    """https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L661"""
    return torch.empty_like(x, dtype=x.dtype) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=x.dtype)


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
):
    """https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/nms.py#L13"""
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv11 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    extra = prediction.shape[1] - nc - 4  # number of extra info
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]  # to track idxs

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # print(f'prediction {prediction.shape}')
    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.zeros((0, 1), device=prediction.device)] * bs  # to store the kept idxs
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        filt = xc[xi]  # confidence
        x = x[filt]
        if return_idxs:
            xk = xk[filt]

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + extra + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, extra), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            if return_idxs:
                xk = xk[i]
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            if return_idxs:
                xk = xk[filt]

        # Filter by class
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            filt = x[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = TorchNMS.fast_nms(boxes, scores, iou_thres, iou_func=batch_probiou)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            # Speed strategy: torchvision for val or already loaded (faster), TorchNMS for predict (lower latency)
            if "torchvision" in sys.modules:
                import torchvision  # scope as slow import

                i = torchvision.ops.nms(boxes, scores, iou_thres)
            else:
                i = TorchNMS.nms(boxes, scores, iou_thres)
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        if (time.time() - t) > time_limit:
            print(f"NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return (output, keepi) if return_idxs else output


class TorchNMS:
    @staticmethod
    def fast_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        iou_threshold: float,
        use_triu: bool = True,
        iou_func=box_iou,
        exit_early: bool = True,
    ):
        """https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/nms.py#L169"""
        if boxes.numel() == 0 and exit_early:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        sorted_idx = torch.argsort(scores, descending=True)
        boxes = boxes[sorted_idx]
        ious = iou_func(boxes, boxes)
        if use_triu:
            ious = ious.triu_(diagonal=1)
            # NOTE: handle the case when len(boxes) hence exportable by eliminating if-else condition
            pick = torch.nonzero((ious >= iou_threshold).sum(0) <= 0).squeeze_(-1)
        else:
            n = boxes.shape[0]
            row_idx = torch.arange(n, device=boxes.device).view(-1, 1).expand(-1, n)
            col_idx = torch.arange(n, device=boxes.device).view(1, -1).expand(n, -1)
            upper_mask = row_idx < col_idx
            ious = ious * upper_mask
            # Zeroing these scores ensures the additional indices would not affect the final results
            scores_ = scores[sorted_idx]
            scores_[~((ious >= iou_threshold).sum(0) <= 0)] = 0
            scores[sorted_idx] = scores_  # update original tensor for NMSModel
            # NOTE: return indices with fixed length to avoid TFLite reshape error
            pick = torch.topk(scores_, scores_.shape[0]).indices
        return sorted_idx[pick]

    @staticmethod
    def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        # Pre-allocate and extract coordinates once
        x1, y1, x2, y2 = boxes.unbind(1)
        areas = (x2 - x1) * (y2 - y1)

        # Sort by scores descending
        order = scores.argsort(0, descending=True)

        # Pre-allocate keep list with maximum possible size
        keep = torch.zeros(order.numel(), dtype=torch.int64, device=boxes.device)
        keep_idx = 0
        while order.numel() > 0:
            i = order[0]
            keep[keep_idx] = i
            keep_idx += 1

            if order.numel() == 1:
                break
            # Vectorized IoU calculation for remaining boxes
            rest = order[1:]
            xx1 = torch.maximum(x1[i], x1[rest])
            yy1 = torch.maximum(y1[i], y1[rest])
            xx2 = torch.minimum(x2[i], x2[rest])
            yy2 = torch.minimum(y2[i], y2[rest])

            # Fast intersection and IoU
            w = (xx2 - xx1).clamp_(min=0)
            h = (yy2 - yy1).clamp_(min=0)
            inter = w * h
            # Early exit: skip IoU calculation if no intersection
            if inter.sum() == 0:
                # No overlaps with current box, keep all remaining boxes
                order = rest
                continue
            iou = inter / (areas[i] + areas[rest] - inter)
            # Keep boxes with IoU <= threshold
            order = rest[iou <= iou_threshold]

        return keep[:keep_idx]

    @staticmethod
    def batched_nms(
        boxes: torch.Tensor,
        scores: torch.Tensor,
        idxs: torch.Tensor,
        iou_threshold: float,
        use_fast_nms: bool = False,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)

        # Strategy: offset boxes by class index to prevent cross-class suppression
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

        return (
            TorchNMS.fast_nms(boxes_for_nms, scores, iou_threshold)
            if use_fast_nms
            else TorchNMS.nms(boxes_for_nms, scores, iou_threshold)
        )


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding: bool = True, xywh: bool = False):
    """https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L102"""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad_x  # x padding
        boxes[..., 1] -= pad_y  # y padding
        if not xywh:
            boxes[..., 2] -= pad_x  # x padding
            boxes[..., 3] -= pad_y  # y padding
    boxes[..., :4] /= gain
    return boxes if xywh else clip_boxes(boxes, img0_shape)


def clip_boxes(boxes, shape):
    """https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py#L152"""
    h, w = shape[:2]  # supports both HWC or HW shapes
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, w)  # x1
        boxes[..., 1].clamp_(0, h)  # y1
        boxes[..., 2].clamp_(0, w)  # x2
        boxes[..., 3].clamp_(0, h)  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)  # y1, y2
    return boxes


def make_anchors(feats_shapes, strides, grid_cell_offset=0.5):
    """
    Generate anchor points and stride tensor, matching the ultralytics Detect head.

    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tal.py#L304

    Args:
        feats_shapes (List[Tuple[int, int]]): Feature map shapes (h, w) per scale, same order as the model outputs.
        strides (List[int]): Stride for each feature map (e.g. [8, 16, 32]).
        grid_cell_offset (float): Grid cell center offset.

    Returns:
        (np.ndarray): Anchor points of shape (sum(h*w), 2) as (x, y).
        (np.ndarray): Stride tensor of shape (sum(h*w), 1).
    """
    anchor_points, stride_tensor = [], []
    for (h, w), stride in zip(feats_shapes, strides):
        sx = np.arange(w, dtype=np.float32) + grid_cell_offset
        sy = np.arange(h, dtype=np.float32) + grid_cell_offset
        gy, gx = np.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(np.stack((gx.ravel(), gy.ravel()), axis=1))
        stride_tensor.append(np.full((h * w, 1), stride, dtype=np.float32))
    return np.concatenate(anchor_points, 0), np.concatenate(stride_tensor, 0)


def softmax(x, axis):
    """Numerically stable softmax along the given axis."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class YOLOv11:
    """
    YOLOv11 object detection model class for handling inference and visualization.

    This class provides functionality to load a YOLOv11 ONNX model, perform inference on images,
    and visualize the detection results.

    Attributes:
        onnx_model (str): Path to the ONNX model file.
        input_image (str): Path to the input image file.
        confidence_thres (float): Confidence threshold for filtering detections.
        iou_thres (float): IoU threshold for non-maximum suppression.
        classes (List[str]): List of class names from the COCO dataset.
        color_palette (np.ndarray): Random color palette for visualizing different classes.
        input_width (int): Width dimension of the model input.
        input_height (int): Height dimension of the model input.
        img (np.ndarray): The loaded input image.
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.
    """

    def __init__(self, onnx_model: str, confidence_thres: float, iou_thres: float):
        """
        Initialize an instance of the YOLOv11 class.

        Args:
            onnx_model (str): Path to the ONNX model.
            input_image (str): Path to the input image.
            confidence_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IoU threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        self.classes = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",5: "bus",
            6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant", 
            11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
            16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
            21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",25: "umbrella",
            26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis",
            31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
            36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
            41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 
            46: "banana", 47: "apple", 48: "sandwich",49: "orange", 50: "broccoli",
            51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 
            56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
            61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote",
            66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster",
            71: "sink", 72: "refrigerator", 73: "book", 74: "clock", 75: "vase",
            76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
        }
        self.cls_map = coco80_to_coco91_class()

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        # Create an inference session using the ONNX model and specify execution providers
        # session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        # axmodel
        self.session = ort_ax.InferenceSession(self.onnx_model, providers=["AxEngineExecutionProvider"])
        
        # Get the model inputs
        # model_inputs = session.get_inputs()

        # Store the shape of the input for later use
        input_shape = [1,3,640,640]
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Precompute anchor points / strides for the native Detect-head decode
        # (replaces the external yolo11s-postprocss.onnx). The feature-map order
        # must match the axmodel output order: stride 8 (80x80), 16 (40x40), 32 (20x20).
        self.strides = (8, 16, 32)
        self.feats_shapes = [(self.input_height // s, self.input_width // s) for s in self.strides]
        self.anchor_points, self.stride_tensor = make_anchors(self.feats_shapes, self.strides)
        self.boxes_json = []


    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (Tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

    def draw_detections(self, img: np.ndarray, box: List[float], score: float, class_id: int) -> None:
        """
        Draw bounding boxes and labels on the input image based on the detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            box (List[float]): Detected bounding box coordinates [x, y, width, height].
            score (float): Confidence score of the detection.
            class_id (int): Class ID for the detected object.
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, input_image) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess the input image before performing inference.

        This method reads the input image, converts its color space, applies letterboxing to maintain aspect ratio,
        normalizes pixel values, and prepares the image data for model input.

        Returns:
            (np.ndarray): Preprocessed image data ready for inference with shape (1, 3, height, width).
            (Tuple[int, int]): Padding values (top, left) applied during letterboxing.
        """
        # Read the input image using OpenCV
        img = cv2.imread(input_image)
        org_img = img.copy()
        # Get the height and width of the input image
        img_height, img_width = img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img, pad = self.letterbox(img, (self.input_width, self.input_height))
        # cv2.imwrite("./output/input.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        image_name = input_image.split("/")[-1].split(".")[0]
        others = {
            "org_img": org_img,
            "image_shape": np.array((self.input_height, self.input_width)),
            "origin_shape": np.array((img_height, img_width)),
            "image_name": image_name,
        }

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0
        # image_data = np.array(img)
         

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data, pad, others

    def postprocess(self, output: List[np.ndarray], pad: Tuple[int, int], other) -> np.ndarray:
        """
        Perform post-processing on the model's output to extract and visualize detections.

        This method processes the raw model output to extract bounding boxes, scores, and class IDs.
        It applies non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            input_image (np.ndarray): The input image.
            output (List[np.ndarray]): The output arrays from the model.
            pad (Tuple[int, int]): Padding values (top, left) used during letterboxing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        org_img = other['org_img']
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []
        org_shape = org_img.shape
        # Calculate the scaling factors for the bounding box coordinates
        gain = min(self.input_height / org_shape[0], self.input_width / org_shape[1])
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]
        # print(f'gain {gain}')

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]
            # print(classes_scores)
            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)
            # print(f'max_score {max_score}')
            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        # print(f'indices {indices}')
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            # print(f'score {score}, class_id {class_id}, box {box}')
            
            box_json = {}
            box_json["image_id"] = int(other['image_name'])
            box_json["category_id"] = self.cls_map[int(class_id)]
            box_json["bbox"] = [round(float(x), 3) for x in box]
            box_json["score"] = round(float(score), 5)
            self.boxes_json.append(box_json)
            
            # Draw the detection on the input image
            self.draw_detections(org_img, box, score, class_id)
        # Return the modified input image
        # cv2.imwrite("./output/ax-output.jpg", org_img)

    def postprocessTroch(self, output, other):
        # print(len(output))
        # print(output[0].shape)
        output_data = torch.from_numpy(output[0])

        preds = non_max_suppression(
            output_data,
            self.confidence_thres,
            self.iou_thres,
            nc=0,
            multi_label=True,
            agnostic=False,
            max_det=300,
            end2end=False,
            rotated=False,
        )
        
        for j, pred in enumerate(preds):
            boxes = pred[:, :4]
            confes = pred[:, -2]
            classes = pred[:, -1]

            boxes = scale_boxes(other['image_shape'], boxes, other['origin_shape'])
            """https://github.com/ultralytics/ultralytics/blob/e3a987c2a74c4ab55e0fbef3b0d8f993cc91c198/ultralytics/models/yolo/detect/val.py#L385"""
            boxes = xyxy2xywh(boxes)  # x,y,x,y->cx,cy,w,h
            boxes[:, :2] -= boxes[:, 2:] / 2  # cx,cy,w,h -> x,y,w,h
            for k, output in enumerate(zip(boxes.tolist(), confes.tolist(), classes.tolist())):
                box, conf, id = output
                box_json = {}
                box_json["image_id"] = int(other['image_name'])
                box_json["category_id"] = self.cls_map[int(id)]
                box_json["bbox"] = [round(float(x), 3) for x in box]
                box_json["score"] = round(conf, 5)
                self.boxes_json.append(box_json)

    def decode_predictions(self, outputs_arr: np.ndarray, reg_max: int = 16) -> List[np.ndarray]:
        """
        Native YOLO11 Detect-head decode, replacing ``self.session_postprocess.run``.

        Takes the concatenated raw head outputs and applies DFL (distribution focal loss) integral,
        anchor-based bbox decoding and class sigmoid, producing the same tensor the original
        ``yolo11s-postprocss.onnx`` returned.

        Args:
            outputs_arr (np.ndarray): Concatenated raw head outputs of shape (1, 4*reg_max + nc, 8400),
                with the box-distribution channels first and the class channels last.
            reg_max (int): DFL bins per box side (16 for YOLO11).

        Returns:
            (List[np.ndarray]): ``[predictions]`` of shape (1, 4 + nc, 8400) in (cx, cy, w, h, *cls)
                format (pixel coordinates in the letterboxed input space), matching the postprocess ONNX.
        """
        no = outputs_arr.shape[1]
        nc = no - 4 * reg_max
        box = outputs_arr[:, : 4 * reg_max, :]  # (1, 64, 8400)
        cls = outputs_arr[:, 4 * reg_max :, :]  # (1, nc, 8400)

        # DFL: softmax over the reg_max bins then expectation with projection [0..reg_max-1]
        b, _, a = box.shape
        box = box.reshape(b, 4, reg_max, a)
        box = softmax(box, axis=2)
        proj = np.arange(reg_max, dtype=np.float32)
        dist = np.tensordot(box, proj, axes=([2], [0]))  # (1, 4, 8400) -> ltrb distances

        # dist2bbox (xywh) + multiply by stride
        anchors = self.anchor_points.T[None]  # (1, 2, 8400)
        stride = self.stride_tensor.reshape(1, 1, -1)  # (1, 1, 8400)
        lt, rb = dist[:, :2, :], dist[:, 2:, :]
        x1y1 = anchors - lt
        x2y2 = anchors + rb
        cxy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        dbox = np.concatenate((cxy, wh), axis=1) * stride  # (1, 4, 8400)

        cls = 1.0 / (1.0 + np.exp(-cls))  # sigmoid
        preds = np.concatenate((dbox, cls), axis=1).astype(np.float32)  # (1, 4 + nc, 8400)
        return [preds]

    def main(self, input_image: str) -> np.ndarray:
        """
        Perform inference using an ONNX model and return the output image with drawn detections.

        Returns:
            (np.ndarray): The output image with drawn detections.
        """

        # Preprocess the image data
        img_data, pad, other = self.preprocess(input_image)

        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {'x': img_data})
        
        out_h = 20*20+40*40+80*80
        outputs_arr = np.zeros((1, 144, out_h))
        start_i = 0
        for k in [0,1,2]:
            out = outputs[k].reshape(1, 144, -1)
            # print(out.shape)
            cur_i  = out.shape[-1] 
            outputs_arr[:, :,start_i:(start_i + cur_i)] = out
            start_i += cur_i
        outputs_arr = outputs_arr.astype(np.float32)
        outputs_post = self.decode_predictions(outputs_arr)
        return self.postprocessTroch(outputs_post, other)  # output image

if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./output-qat11/compiled.axmodel", help="Input your ONNX model.")
    parser.add_argument("--img_dir", type=str, default='/root/data/', help="Path to input image dir.")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.7, help="NMS IoU threshold")
    args = parser.parse_args()

    # Create an instance of the YOLOv11 class with the specified arguments
    detection = YOLOv11(args.model, args.conf_thres, args.iou_thres)
    
    def find_images_and_targets(folder):
        images = sorted(smart_glob(os.path.join(folder, "*.jpg")))
        return images

    images = find_images_and_targets(args.img_dir)
    print(len(images))
    for img_p in tqdm.tqdm(images):
        detection.main(img_p)
        # break

    tmp_preds_path = "./ax_preds-qat11.json"
    with open(tmp_preds_path, "w") as f:
        json.dump(detection.boxes_json, f, ensure_ascii=False, indent=2)
