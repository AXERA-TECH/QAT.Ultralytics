"""Tests for the shared YOLO26/YOLO11 raw detection-output decoder."""

import importlib.util
from pathlib import Path

import numpy as np


def _load_detection_module():
    path = Path(__file__).resolve().parents[1] / "axera-npu" / "run_yolo_detect.py"
    spec = importlib.util.spec_from_file_location("run_yolo_detect", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_regression_supports_yolo26_distances_and_yolo11_dfl_logits():
    decoder = _load_detection_module()

    distances = np.array([[[1.0], [2.0], [3.0], [4.0]]], dtype=np.float32)
    assert np.array_equal(decoder.decode_regression(distances), np.array([[1.0, 2.0, 3.0, 4.0]]))

    dfl = np.full((1, 64, 1), -20.0, dtype=np.float32)
    for side in range(4):
        dfl[0, side * 16 + 2, 0] = 20.0
    assert np.allclose(decoder.decode_regression(dfl), np.full((1, 4), 2.0), atol=1e-5)


def test_detection_output_pairing_ignores_yolo11_export_output_order():
    decoder = _load_detection_module()
    outputs = []
    for anchors in (64, 16, 4):
        regression = np.full((1, 64, anchors), -20.0, dtype=np.float32)
        for side in range(4):
            regression[0, side * 16 + 2] = 20.0
        outputs.append(regression)
    for anchors in (64, 16, 4):
        scores = np.full((1, 80, anchors), -20.0, dtype=np.float32)
        scores[0, 7, 0] = 20.0
        outputs.append(scores)

    predictions = decoder.decode_yolo_detection(outputs, num_classes=80, max_det=3, input_hw=(64, 64))

    assert predictions.shape == (3, 6)
    assert predictions[:, 5].tolist() == [7.0, 7.0, 7.0]
    assert np.allclose(predictions[:, 4], 1.0, atol=1e-5)


def test_detection_decode_supports_rectangular_input():
    decoder = _load_detection_module()
    outputs = []
    for anchors in (96, 24, 6):
        boxes = np.zeros((1, 4, anchors), dtype=np.float32)
        scores = np.full((1, 80, anchors), -20.0, dtype=np.float32)
        scores[0, 7, 0] = 20.0
        outputs.extend((boxes, scores))

    predictions = decoder.decode_yolo_detection(
        outputs, num_classes=80, max_det=1, input_hw=(64, 96), head_type="one2one"
    )

    assert predictions.shape == (1, 6)
    assert np.allclose(predictions[0, :4], [4.0, 4.0, 4.0, 4.0])
    assert predictions[0, 5] == 7.0


def test_yolo11_nms_removes_overlapping_boxes_of_the_same_class():
    decoder = _load_detection_module()
    boxes = np.array(
        [
            [10.0, 10.0, 110.0, 110.0],
            [12.0, 12.0, 112.0, 112.0],
            [10.0, 10.0, 110.0, 110.0],
        ],
        dtype=np.float32,
    )
    scores = np.zeros((3, 80), dtype=np.float32)
    scores[0, 5] = 0.95
    scores[1, 5] = 0.93
    scores[2, 6] = 0.94

    predictions = decoder.class_aware_nms(boxes, scores, conf_thres=0.25, iou_thres=0.7, max_det=300)

    assert predictions.shape == (2, 6)
    assert predictions[:, 5].tolist() == [5.0, 6.0]
    assert np.allclose(predictions[:, 4], [0.95, 0.94])


def test_yolo26_named_one2many_outputs_use_nms(monkeypatch):
    decoder = _load_detection_module()
    outputs = []
    for anchors in (144, 36, 9):
        outputs.append(np.zeros((1, 4, anchors), dtype=np.float32))
        outputs.append(np.zeros((1, 80, anchors), dtype=np.float32))

    called = {}

    def fake_nms(boxes, scores, conf_thres, iou_thres, max_det):
        called["shapes"] = boxes.shape, scores.shape
        return np.empty((0, 6), dtype=np.float32)

    monkeypatch.setattr(decoder, "class_aware_nms", fake_nms)
    decoder.decode_yolo_detection(
        outputs,
        num_classes=80,
        max_det=300,
        input_hw=(96, 96),
        output_names=["boxes_p3", "scores_p3", "boxes_p4", "scores_p4", "boxes_p5", "scores_p5"],
    )

    assert called["shapes"] == ((189, 4), (189, 80))


def test_detection_decode_rejects_anchor_count_mismatch():
    decoder = _load_detection_module()
    outputs = []
    for anchors in (95, 24, 6):
        outputs.append(np.zeros((1, 4, anchors), dtype=np.float32))
        outputs.append(np.zeros((1, 80, anchors), dtype=np.float32))

    with np.testing.assert_raises_regex(ValueError, "Anchor count mismatch at stride 8"):
        decoder.decode_yolo_detection(outputs, num_classes=80, max_det=1, input_hw=(64, 96))
