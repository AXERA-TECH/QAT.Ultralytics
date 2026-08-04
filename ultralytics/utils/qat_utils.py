from __future__ import annotations

from pathlib import Path

import torch
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
from torch.export import Dim

from ultralytics.utils.ax_quantizer import AXQuantizer, ax_load_config

LEGACY_QAT_CONFIG_NAMES = {
    "config_exp57_attn_s8.json": "config_siluInU16_attnS8_clsU16.json",
    "config_exp58_silu_u8_attn_s8.json": "config_siluInU8_attnS8_clsU16.json",
}


def resolve_qat_config_path(config_path: str | Path) -> Path:
    """Resolve current and legacy QAT config paths from CLI arguments or checkpoint metadata."""
    path = Path(config_path)
    current_name = LEGACY_QAT_CONFIG_NAMES.get(path.name, path.name)
    candidates = [path]
    if current_name != path.name:
        candidates.append(path.with_name(current_name))
    candidates.append(Path("config-qat") / current_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return Path("config-qat") / current_name


def _load_quantizer(use_lsq: bool = False):
    """Load the appropriate quantizer module."""
    if use_lsq:
        from ultralytics.utils.ax_quantizer_lsq import AXQuantizer as LSQQuantizer
        from ultralytics.utils.ax_quantizer_lsq import ax_load_config as lsq_load_config

        return LSQQuantizer, lsq_load_config
    return AXQuantizer, ax_load_config


def _normalize_imgsz(imgsz: int | list[int] | tuple[int, int]) -> tuple[int, int]:
    """Normalize Ultralytics `imgsz` config to a fixed `(height, width)` tuple."""
    if isinstance(imgsz, int):
        return imgsz, imgsz
    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 1:
            return int(imgsz[0]), int(imgsz[0])
        return int(imgsz[0]), int(imgsz[1])
    raise TypeError(f"Unsupported imgsz type: {type(imgsz).__name__}")


def prepare_pt2e_qat_model(
    float_model: torch.nn.Module,
    device: torch.device | str,
    config_path: str | Path,
    imgsz: int | list[int] | tuple[int, int],
    dynamic_batch_max: int = 128,
    input_name: str = "x",
    use_lsq: bool = False,
) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
    """Export a training graph and prepare a PT2E QAT model.

    All dimensions (batch, H, W) use ``Dim.AUTO`` so the exported model accepts variable spatial sizes (needed for
    ``rect=True`` validation and deployment).
    """
    height, width = _normalize_imgsz(imgsz)
    max_batch = max(int(dynamic_batch_max), 2)
    example_batch = min(2, max_batch)
    config_path = resolve_qat_config_path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"QAT config file not found: {config_path}")

    QuantizerClass, load_config = _load_quantizer(use_lsq)

    inputs = torch.rand(example_batch, 3, height, width, device=device).contiguous()
    global_config, regional_configs = load_config(str(config_path))
    quantizer = QuantizerClass()
    quantizer.set_global(global_config)
    quantizer.set_regional(regional_configs)

    exported_program = torch.export.export_for_training(
        float_model,
        (inputs,),
        dynamic_shapes={input_name: {0: Dim.AUTO, 2: Dim.AUTO, 3: Dim.AUTO}},
    )
    exported_model = exported_program.module().to(device)
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    torch.ao.quantization.move_exported_model_to_eval(prepared_model)
    torch.ao.quantization.allow_exported_model_train_eval(prepared_model)
    return exported_model, prepared_model.to(device)
