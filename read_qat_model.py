from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e

from ultralytics import YOLO
from ultralytics.utils.ax_quantizer import AXQuantizer, ax_load_config


def format_source_fn_stack(node: torch.fx.Node) -> str:
    """Return a compact source_fn_stack string for easier module mapping."""
    source_fn_stack = node.meta.get("source_fn_stack")
    if not source_fn_stack:
        return "-"
    parts = []
    for name, fn in source_fn_stack:
        fn_name = getattr(fn, "__name__", repr(fn))
        parts.append(f"{name}:{fn_name}")
    return " | ".join(parts)


def dump_graph(graph_module: torch.fx.GraphModule, output_path: Path, prefix: str | None) -> int:
    """Write graph node info to file and return matched node count."""
    lines = []
    matched = 0
    for node in graph_module.graph.nodes:
        if prefix and not node.name.startswith(prefix):
            continue
        matched += 1
        lines.append(f"name={node.name}")
        lines.append(f"op={node.op}")
        lines.append(f"target={node.target}")
        lines.append(f"source_fn_stack={format_source_fn_stack(node)}")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return matched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump FX graph node names for QAT config selection.")
    parser.add_argument("--model", default="yolo11n.yaml", help="YOLO model yaml or checkpoint path.")
    parser.add_argument("--weights", default="yolo11n.pt", help="Optional float pretrained weights.")
    parser.add_argument("--config", default="./config.json", help="Quantizer config JSON path.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:2.")
    parser.add_argument("--imgsz", nargs=2, type=int, default=[640, 640], metavar=("H", "W"))
    parser.add_argument(
        "--prefix",
        default="conv2d_",
        help="Only dump nodes whose FX node.name starts with this prefix. Use empty string for all nodes.",
    )
    parser.add_argument(
        "--output-dir",
        default="./debug/read_qat_model",
        help="Directory to save exported/prepared graph dumps.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Only export float FX graph and skip prepare_qat_pt2e.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    prefix = args.prefix or None

    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)

    float_model = model.model.to(device)
    inputs = torch.rand(1, 3, *args.imgsz).to(device)
    print(f"export input shape: {tuple(inputs.shape)}")
    exported_model = torch.export.export_for_training(float_model, (inputs,), dynamic_shapes=None)
    exported_module = exported_model.module()

    exported_txt = output_dir / "exported_graph.txt"
    exported_count = dump_graph(exported_module, exported_txt, prefix)
    print(f"saved exported graph nodes to {exported_txt} (matched={exported_count})")

    if args.skip_prepare:
        return

    global_config, regional_configs = ax_load_config(args.config)
    quantizer = AXQuantizer()
    quantizer.set_global(global_config)
    quantizer.set_regional(regional_configs)

    prepared_model = prepare_qat_pt2e(exported_module, quantizer)
    prepared_txt = output_dir / "prepared_graph.txt"
    prepared_count = dump_graph(prepared_model, prepared_txt, prefix)
    print(f"saved prepared graph nodes to {prepared_txt} (matched={prepared_count})")


if __name__ == "__main__":
    main()
