from copy import deepcopy

import cv2
import numpy as np
import pytest

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG


def _make_dataset(tmp_path, *, imgsz, augment=False, rect=False):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    cv2.imwrite(str(images / "sample.jpg"), np.zeros((120, 40, 3), dtype=np.uint8))
    (labels / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    return YOLODataset(
        img_path=str(images),
        imgsz=imgsz,
        batch_size=1,
        augment=augment,
        hyp=deepcopy(DEFAULT_CFG),
        rect=rect,
        cache=False,
        stride=32,
        task="detect",
        data={"names": {0: "object"}, "channels": 3},
    )


def test_validation_dataset_supports_fixed_rectangular_shape(tmp_path):
    dataset = _make_dataset(tmp_path, imgsz=(64, 96))

    sample = dataset[0]

    assert tuple(sample["img"].shape) == (3, 64, 96)
    assert sample["resized_shape"] == (64, 96)


@pytest.mark.parametrize("kwargs", [{"augment": True}, {"rect": True}])
def test_two_dimensional_imgsz_is_rejected_outside_fixed_validation(tmp_path, kwargs):
    with pytest.raises(ValueError, match="fixed-shape validation"):
        _make_dataset(tmp_path, imgsz=(64, 96), **kwargs)
