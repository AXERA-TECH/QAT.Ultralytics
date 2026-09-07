import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_test_module():
    spec = importlib.util.spec_from_file_location("qat_test_cli", ROOT / "test.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (640, (640, 640)),
        ([640], (640, 640)),
        ([352, 640], (352, 640)),
        ((640, 352), (640, 352)),
    ],
)
def test_normalize_imgsz(value, expected):
    assert load_test_module().normalize_imgsz(value) == expected


@pytest.mark.parametrize("value", [[], [320, 640, 960], [0], [-1, 640]])
def test_normalize_imgsz_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        load_test_module().normalize_imgsz(value)


def test_parse_args_accepts_rectangular_imgsz(monkeypatch):
    module = load_test_module()
    monkeypatch.setattr(sys, "argv", ["test.py", "--imgsz", "352", "640"])
    assert module.normalize_imgsz(module.parse_args().imgsz) == (352, 640)


def test_parse_args_defaults_to_explicit_square_imgsz(monkeypatch):
    module = load_test_module()
    monkeypatch.setattr(sys, "argv", ["test.py"])
    assert module.parse_args().imgsz == [640, 640]


def test_checkpoint_backend_prepares_rectangular_graph(monkeypatch, tmp_path):
    module = load_test_module()
    config = tmp_path / "config.json"
    config.touch()
    prepared = SimpleNamespace(
        load_state_dict=lambda state, strict=False: SimpleNamespace(missing_keys=[], unexpected_keys=[]),
        apply=lambda function: None,
        to=lambda device: prepared,
    )
    reference = SimpleNamespace(train=lambda: reference, eval=lambda: reference)
    captured = {}

    def prepare(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(), prepared

    monkeypatch.setattr(module.torch, "load", lambda *args, **kwargs: {"qat_model": {"weight": 1}})
    monkeypatch.setattr(module, "resolve_qat_config_path", lambda path: config)
    monkeypatch.setattr(module, "prepare_pt2e_qat_model", prepare)
    monkeypatch.setattr(module.BaseValidator, "_prepare_pt2e_model_for_eval", lambda model: model)
    args = SimpleNamespace(
        model="best.pt", quant_config=str(config), imgsz=(352, 640), task="detect"
    )

    module.QATCheckpointBackend(args, reference, module.torch.device("cpu"))

    assert captured["imgsz"] == (352, 640)
