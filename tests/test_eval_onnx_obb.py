from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.eval_backends.onnx_obb import OrtOBB


class _Session:
    def __init__(self, outputs, names=("boxes", "scores", "angle"), input_hw=(64, 64)):
        self.outputs = outputs
        self.input_hw = input_hw
        self.input = SimpleNamespace(name="images", shape=[1, 3, *input_hw])
        self.output_meta = [SimpleNamespace(name=name) for name in names]

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return self.output_meta

    def run(self, _output_names, feed):
        assert feed["images"].shape == (1, 3, *self.input_hw)
        return self.outputs


def _head():
    return SimpleNamespace(reg_max=1, nc=15, ne=1, stride=torch.tensor([8.0, 16.0, 32.0]))


def _outputs(anchors=84):
    return [
        np.zeros((1, 4, anchors), dtype=np.float32),
        np.zeros((1, 15, anchors), dtype=np.float32),
        np.zeros((1, 1, anchors), dtype=np.float32),
    ]


def test_ort_obb_rebuilds_batched_pt2e_prediction_contract():
    wrapper = OrtOBB("unused.onnx", torch.device("cpu"), _head(), session=_Session(_outputs()))

    predictions = wrapper(torch.zeros(2, 3, 64, 64))

    assert set(predictions) == {"one2one", "one2many"}
    one2one = predictions["one2one"]
    assert one2one["boxes"].shape == (2, 4, 84)
    assert one2one["scores"].shape == (2, 15, 84)
    assert one2one["angle"].shape == (2, 1, 84)
    assert [tuple(feat.shape) for feat in one2one["feats"]] == [
        (2, 1, 8, 8),
        (2, 1, 4, 4),
        (2, 1, 2, 2),
    ]


def test_ort_obb_rejects_mismatched_anchor_counts():
    outputs = _outputs()
    outputs[-1] = np.zeros((1, 1, 80), dtype=np.float32)
    wrapper = OrtOBB("unused.onnx", torch.device("cpu"), _head(), session=_Session(outputs))

    with pytest.raises(RuntimeError, match="same anchor count"):
        wrapper(torch.zeros(1, 3, 64, 64))


def test_ort_obb_can_classify_outputs_without_stable_names():
    outputs = _outputs()
    session = _Session([outputs[2], outputs[0], outputs[1]], names=("output0", "output1", "output2"))
    wrapper = OrtOBB("unused.onnx", torch.device("cpu"), _head(), session=session)

    predictions = wrapper(torch.zeros(1, 3, 64, 64))

    assert predictions["one2one"]["boxes"].shape[1] == 4
    assert predictions["one2one"]["scores"].shape[1] == 15
    assert predictions["one2one"]["angle"].shape[1] == 1


def test_ort_obb_rejects_wrong_spatial_input():
    wrapper = OrtOBB("unused.onnx", torch.device("cpu"), _head(), session=_Session(_outputs()))

    with pytest.raises(RuntimeError, match="expects spatial input"):
        wrapper(torch.zeros(1, 3, 32, 32))


def test_ort_obb_supports_rectangular_input():
    session = _Session(_outputs(126), input_hw=(64, 96))
    wrapper = OrtOBB("unused.onnx", torch.device("cpu"), _head(), session=session)

    predictions = wrapper(torch.zeros(2, 3, 64, 96))

    assert [tuple(feat.shape) for feat in predictions["one2one"]["feats"]] == [
        (2, 1, 8, 12),
        (2, 1, 4, 6),
        (2, 1, 2, 3),
    ]
