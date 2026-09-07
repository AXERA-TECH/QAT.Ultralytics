"""Shared argument helpers for fixed-shape ONNX evaluators."""

from __future__ import annotations


def normalize_imgsz(values: int | list[int] | tuple[int, ...] | None) -> tuple[int, int] | None:
    """Normalize one or two CLI image-size values to ``(height, width)``."""
    if values is None:
        return None
    values = [values] if isinstance(values, int) else list(values)
    if len(values) == 1:
        height = width = int(values[0])
    elif len(values) == 2:
        height, width = (int(value) for value in values)
    else:
        raise ValueError(f"--imgsz expects one value or height width, got {values}")
    if height <= 0 or width <= 0:
        raise ValueError(f"--imgsz values must be positive, got {height}x{width}")
    return height, width


def resolve_imgsz(values: int | list[int] | tuple[int, ...] | None, input_hw: tuple[int, int]) -> tuple[int, int]:
    """Use the ONNX fixed input by default and reject a mismatching explicit size."""
    requested = normalize_imgsz(values)
    input_hw = tuple(int(value) for value in input_hw)
    if requested is not None and requested != input_hw:
        raise ValueError(f"--imgsz {requested} does not match QuantONNX input {input_hw}")
    return input_hw
