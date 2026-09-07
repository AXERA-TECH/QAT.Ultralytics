"""Tests for the AXERA YOLO26 segmentation-output decoder."""

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_segmentation_module():
    path = Path(__file__).resolve().parents[1] / "axera-npu" / "run_yolo_seg.py"
    spec = importlib.util.spec_from_file_location("run_yolo_seg", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_segmentation_decode_supports_rectangular_input():
    decoder = _load_segmentation_module()
    anchor_counts = (96, 24, 6)
    boxes = [np.zeros((1, 4, count), dtype=np.float32) for count in anchor_counts]
    scores = [np.full((1, 80, count), -20.0, dtype=np.float32) for count in anchor_counts]
    scores[0][0, 7, 0] = 20.0
    mask_coefficients = np.zeros((1, 32, sum(anchor_counts)), dtype=np.float32)

    predictions = decoder.decode_segment_outputs(
        boxes, scores, mask_coefficients, imgsz=(64, 96), max_det=1
    )

    assert predictions.shape == (1, 38)
    assert np.allclose(predictions[0, :4], [4.0, 4.0, 4.0, 4.0])
    assert predictions[0, 5] == 7.0
