#!/usr/bin/env python3
"""Unified evaluation entrypoint for float, QAT, converted, ONNX, segmentation, OBB, and PTQ models."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

BACKENDS = {
    "float": "float.py",
    "qat": "qat.py",
    "convert": "convert.py",
    "onnx": "onnx.py",
    "onnx-obb": "onnx_obb.py",
    "onnx-pose": "onnx_pose.py",
    "segment": "segment.py",
    "ptq": "ptq.py",
    "onnx-one2many": "onnx_one2many.py",
}


def _print_help() -> None:
    print(
        "Usage: python eval.py <mode> [options]\n\n"
        "Modes:\n"
        "  float          Evaluate a float PyTorch model with YOLO.val\n"
        "  qat            Evaluate a detect/OBB/Pose/Classify QAT checkpoint with fake quantization\n"
        "  convert        Evaluate a detect/OBB/Pose/Classify QAT checkpoint after convert_pt2e (real Q/DQ)\n"
        "  onnx           Evaluate a six-output one2one QAT ONNX model with ORT\n"
        "  onnx-obb       Evaluate a three-output OBB QuantONNX model with ORT\n"
        "  onnx-pose      Evaluate a three-output Pose QuantONNX model with ORT\n"
        "  segment        Evaluate a QAT segmentation checkpoint\n"
        "  ptq            Calibrate, convert, and evaluate a PTQ model\n"
        "  onnx-one2many  Evaluate a legacy one2many ONNX model with NMS\n\n"
        "Run 'python eval.py <mode> --help' for mode-specific options."
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _print_help()
        return

    mode = sys.argv[1]
    backend = BACKENDS.get(mode)
    if backend is None:
        choices = ", ".join(BACKENDS)
        raise SystemExit(f"Unknown eval mode '{mode}'. Choose one of: {choices}")

    entrypoint = Path(__file__).resolve().parent / "scripts" / "eval_backends" / backend
    sys.argv = [f"eval.py {mode}", *sys.argv[2:]]
    module_name = f"_eval_backend_{mode.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load eval backend: {entrypoint}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    backend_main = getattr(module, "main", None)
    if backend_main is not None:
        backend_main()


if __name__ == "__main__":
    main()
