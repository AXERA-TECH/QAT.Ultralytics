from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.eval_backends.onnx import OrtOne2One
from scripts.eval_backends.onnx_utils import resolve_imgsz


class _Session:
    def __init__(self, input_hw=(64, 96), anchors=(96, 24, 6)):
        self.input_hw = input_hw
        self.anchors = anchors
        self.input = SimpleNamespace(name="images", shape=[1, 3, *input_hw])

    def get_inputs(self):
        return [self.input]

    def run(self, _names, feed):
        assert feed["images"].shape == (1, 3, *self.input_hw)
        outputs = []
        for anchors in self.anchors:
            outputs.extend(
                (
                    np.zeros((1, 4, anchors), dtype=np.float32),
                    np.zeros((1, 80, anchors), dtype=np.float32),
                )
            )
        return outputs


def test_ort_detect_rebuilds_rectangular_feature_maps():
    wrapper = OrtOne2One(
        "unused.onnx",
        torch.device("cpu"),
        box_channels=4,
        score_channels=80,
        end2end=True,
        strides=(8, 16, 32),
        session=_Session(),
    )

    predictions = wrapper(torch.zeros(2, 3, 64, 96))

    assert [tuple(feat.shape) for feat in predictions["one2one"]["feats"]] == [
        (2, 1, 8, 12),
        (2, 1, 4, 6),
        (2, 1, 2, 3),
    ]


def test_ort_detect_rejects_anchor_count_mismatch():
    wrapper = OrtOne2One(
        "unused.onnx",
        torch.device("cpu"),
        box_channels=4,
        score_channels=80,
        end2end=False,
        strides=(8, 16, 32),
        session=_Session(anchors=(95, 24, 6)),
    )

    with pytest.raises(RuntimeError, match="Anchor count mismatch at stride 8"):
        wrapper(torch.zeros(1, 3, 64, 96))


def test_resolve_imgsz_uses_onnx_shape_and_checks_explicit_override():
    assert resolve_imgsz(None, (64, 96)) == (64, 96)
    assert resolve_imgsz([64, 96], (64, 96)) == (64, 96)
    with pytest.raises(ValueError, match="does not match QuantONNX input"):
        resolve_imgsz([64], (64, 96))
