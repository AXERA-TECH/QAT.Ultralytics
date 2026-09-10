from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.eval_backends.onnx_pose import OrtPose


class _Session:
    def __init__(self, outputs, input_hw=(64, 64)):
        self.outputs = outputs
        self.input_hw = input_hw
        self.input = SimpleNamespace(name="images", shape=[1, 3, *input_hw])
        self.output_meta = [SimpleNamespace(name=name) for name in ("boxes", "scores", "keypoints")]

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return self.output_meta

    def run(self, _names, feed):
        assert feed["images"].shape == (1, 3, *self.input_hw)
        return self.outputs


def _head():
    return SimpleNamespace(reg_max=1, nc=1, nk=51, stride=torch.tensor([8.0, 16.0, 32.0]))


def _outputs(anchors=84):
    return [
        np.zeros((1, 4, anchors), dtype=np.float32),
        np.zeros((1, 1, anchors), dtype=np.float32),
        np.zeros((1, 51, anchors), dtype=np.float32),
    ]


def test_ort_pose_rebuilds_batched_pt2e_contract():
    wrapper = OrtPose("unused.onnx", torch.device("cpu"), _head(), session=_Session(_outputs()))
    predictions = wrapper(torch.zeros(2, 3, 64, 64))

    assert predictions["one2one"]["boxes"].shape == (2, 4, 84)
    assert predictions["one2one"]["scores"].shape == (2, 1, 84)
    assert predictions["one2one"]["kpts"].shape == (2, 51, 84)


def test_ort_pose_rejects_mismatched_anchor_counts():
    outputs = _outputs()
    outputs[-1] = np.zeros((1, 51, 80), dtype=np.float32)
    wrapper = OrtPose("unused.onnx", torch.device("cpu"), _head(), session=_Session(outputs))

    with pytest.raises(RuntimeError, match="same anchor count"):
        wrapper(torch.zeros(1, 3, 64, 64))


def test_ort_pose_supports_rectangular_input():
    session = _Session(_outputs(126), input_hw=(64, 96))
    wrapper = OrtPose("unused.onnx", torch.device("cpu"), _head(), session=session)

    predictions = wrapper(torch.zeros(2, 3, 64, 96))

    assert [tuple(feat.shape) for feat in predictions["one2one"]["feats"]] == [
        (2, 1, 8, 12),
        (2, 1, 4, 6),
        (2, 1, 2, 3),
    ]
