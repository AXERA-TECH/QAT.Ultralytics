import numpy as np

from scripts.eval_backends.onnx_one2many import decode_predictions


def test_legacy_one2many_decode_uses_rectangular_feature_maps():
    outputs = []
    for height, width in ((8, 12), (4, 6), (2, 3)):
        outputs.extend(
            (
                np.zeros((1, height, width, 4), dtype=np.float32),
                np.full((1, height, width, 80), -20.0, dtype=np.float32),
            )
        )

    predictions = decode_predictions(outputs, input_hw=(64, 96))

    assert predictions.shape == (1, 126, 84)
    # Zero ltrb distances decode to the first P3 anchor center at (4, 4).
    assert np.allclose(predictions[0, 0, :4].numpy(), [4.0, 4.0, 4.0, 4.0])


def test_legacy_one2many_decode_supports_rectangular_flattened_outputs():
    outputs = []
    for anchors in (96, 24, 6):
        outputs.extend(
            (
                np.zeros((1, 4, anchors), dtype=np.float32),
                np.full((1, 80, anchors), -20.0, dtype=np.float32),
            )
        )

    predictions = decode_predictions(outputs, input_hw=(64, 96))

    assert predictions.shape == (1, 126, 84)
    assert np.allclose(predictions[0, 95, :4].numpy(), [92.0, 60.0, 92.0, 60.0])
