"""Unit tests for the QAT training entry-point configuration selection."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import train_qat


def _load_train_gpus_module():
    path = Path(__file__).resolve().parents[1] / "train_gpus.py"
    spec = importlib.util.spec_from_file_location("train_gpus", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("module", "expected_suffix"),
    [(train_qat, "-qat"), (_load_train_gpus_module(), "-qat-gpus")],
)
def test_explicit_quant_config_overrides_profile(module, expected_suffix, tmp_path):
    config = tmp_path / "custom.json"
    config.write_text("{}")
    args = SimpleNamespace(model="yolo11n.yaml", profile="throughput", quant_config=str(config))

    resolved, name = module.resolve_config_and_name(args)

    assert resolved == config
    assert name == f"yolo11n{expected_suffix}"


@pytest.mark.parametrize("module", [train_qat, _load_train_gpus_module()])
def test_config_selection_requires_profile_or_explicit_config(module):
    args = SimpleNamespace(model="yolo11n.yaml", profile=None, quant_config=None)

    with pytest.raises(ValueError, match="quant-config"):
        module.resolve_config_and_name(args)


@pytest.mark.parametrize("module", [train_qat, _load_train_gpus_module()])
def test_relative_project_is_resolved_inside_repository(module):
    assert module.resolve_project_dir(None, "detect") == str(module.ROOT / "runs" / "detect")
    assert module.resolve_project_dir("runs/custom", "detect") == str(module.ROOT / "runs/custom")


def test_train_qat_accepts_pose_task():
    parser_args = [
        "--task",
        "pose",
        "--model",
        "yolo26n-pose.yaml",
        "--pretrained",
        "weights/yolo26n-pose.pt",
        "--data",
        "coco8-pose.yaml",
        "--quant-config",
        "config-qat/config.json",
    ]

    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["train_qat.py", *parser_args]
        args = train_qat.parse_args()
    finally:
        sys.argv = old_argv

    assert args.task == "pose"


def test_train_qat_accepts_classify_task():
    parser_args = [
        "--task",
        "classify",
        "--model",
        "yolo26n-cls.yaml",
        "--pretrained",
        "weights/yolo26n-cls.pt",
        "--data",
        "imagenet10",
        "--quant-config",
        "config-qat/config.json",
    ]

    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["train_qat.py", *parser_args]
        args = train_qat.parse_args()
    finally:
        sys.argv = old_argv

    assert args.task == "classify"
