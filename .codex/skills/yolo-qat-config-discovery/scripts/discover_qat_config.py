#!/usr/bin/env python3
"""Discover YOLO FX node roles and regenerate regional QAT module names."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.export import Dim

from ultralytics import YOLO

CONV_OPS = {torch.ops.aten.conv2d.default}
SILU_OPS = {torch.ops.aten.silu.default, torch.ops.aten.silu_.default}
MATMUL_OPS = {torch.ops.aten.matmul.default}
MUL_OPS = {torch.ops.aten.mul.Tensor}
SOFTMAX_OPS = {torch.ops.aten.softmax.int}
TRANSPARENT_OPS = {
    torch.ops.aten.detach.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.contiguous.default,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="YOLO model YAML.")
    parser.add_argument("--pretrained", default=None, help="Optional matching float checkpoint.")
    parser.add_argument("--base-config", required=True, type=Path, help="Validated QAT config used as a template.")
    parser.add_argument("--output", type=Path, help="Generated QAT config path.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that the template already matches the current FX graph without writing a config.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Optional discovery report JSON path.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--cls-last-n", type=int, default=2)
    parser.add_argument(
        "--cls-u16",
        choices=("auto", "on", "off"),
        default="auto",
        help="Configure optional classification U16 entries; auto follows the template.",
    )
    parser.add_argument(
        "--expected-attention",
        type=int,
        default=None,
        help="Optional strict Attention region count; otherwise configure every complete Attention module.",
    )
    parser.add_argument("--branch", choices=("auto", "one2one_cv3", "cv3"), default="auto")
    return parser.parse_args()


def module_entries(node: torch.fx.Node) -> list[tuple[str, str]]:
    stack = node.meta.get("nn_module_stack") or {}
    entries = []
    for value in stack.values():
        if isinstance(value, tuple) and len(value) >= 2:
            entries.append((str(value[0]), str(value[1])))
    return entries


def module_paths(node: torch.fx.Node) -> list[str]:
    return [path for path, _ in module_entries(node)]


def attention_owner(node: torch.fx.Node) -> str | None:
    for path, module_type in reversed(module_entries(node)):
        if module_type.endswith(".Attention"):
            return path
    return None


def cls_location(node: torch.fx.Node) -> tuple[str, int] | None:
    for path in reversed(module_paths(node)):
        match = re.search(r"(?:^|\.)(one2one_cv3|cv3)\.(\d+)(?:\.|$)", path)
        if match:
            return match.group(1), int(match.group(2))
    return None


def node_args(value: Any):
    if isinstance(value, torch.fx.Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from node_args(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from node_args(item)


def nearest_upstream_silu(start: torch.fx.Node, max_depth: int = 8) -> torch.fx.Node | None:
    queue = [(start, 0)]
    visited = set()
    while queue:
        node, depth = queue.pop(0)
        if node in visited or depth > max_depth:
            continue
        visited.add(node)
        if node.op == "call_function" and node.target in SILU_OPS:
            return node
        if node.op == "call_function" and node.target in CONV_OPS:
            continue
        queue.extend((arg, depth + 1) for arg in node_args((node.args, node.kwargs)))
    return None


def fanout_convs(start: torch.fx.Node, max_depth: int = 4) -> list[torch.fx.Node]:
    queue = [(start, 0)]
    visited = set()
    found = []
    while queue:
        node, depth = queue.pop(0)
        if node in visited or depth > max_depth:
            continue
        visited.add(node)
        for user in node.users:
            if user.op != "call_function":
                continue
            if user.target in CONV_OPS:
                found.append(user)
            elif user.target in TRANSPARENT_OPS:
                queue.append((user, depth + 1))
    return found


def unique_nodes(nodes: list[torch.fx.Node], order: dict[torch.fx.Node, int]) -> list[torch.fx.Node]:
    return sorted(set(nodes), key=order.__getitem__)


def discover_cls_u16(
    graph_nodes: list[torch.fx.Node],
    order: dict[torch.fx.Node, int],
    branch_arg: str,
    cls_last_n: int,
) -> dict[str, Any]:
    cls_by_branch: dict[str, dict[int, list[torch.fx.Node]]] = defaultdict(lambda: defaultdict(list))
    for node in graph_nodes:
        location = cls_location(node)
        if location:
            branch, level = location
            cls_by_branch[branch][level].append(node)

    if branch_arg == "auto":
        branch = "one2one_cv3" if cls_by_branch.get("one2one_cv3") else "cv3"
    else:
        branch = branch_arg
    levels = sorted(cls_by_branch.get(branch, {}))
    if len(levels) < cls_last_n:
        raise RuntimeError(f"{branch} has {len(levels)} level(s), expected at least {cls_last_n}: {levels}")
    selected_levels = levels[-cls_last_n:]

    cls_silus = []
    cls_input_convs = []
    cls_output_convs = []
    first_conv_by_level = {}
    cls_report = []
    for level in selected_levels:
        level_nodes = cls_by_branch[branch][level]
        convs = unique_nodes([node for node in level_nodes if node.target in CONV_OPS], order)
        silus = unique_nodes([node for node in level_nodes if node.target in SILU_OPS], order)
        if len(convs) < 2 or not silus:
            raise RuntimeError(f"Incomplete {branch}.{level}: convs={len(convs)}, silus={len(silus)}")
        first_conv_by_level[level] = convs[0]
        cls_silus.extend(silus)
        cls_input_convs.extend(convs[1:-1])
        cls_output_convs.append(convs[-1])
        cls_report.append(
            {
                "level": level,
                "convs": [node.name for node in convs],
                "silus": [node.name for node in silus],
            }
        )

    highest_level = selected_levels[-1]
    boundary_first_conv = first_conv_by_level[highest_level]
    cls_input_convs.append(boundary_first_conv)
    boundary_input = next(node_args(boundary_first_conv.args[0]), None)
    if boundary_input is None:
        raise RuntimeError(f"No tensor input found for {boundary_first_conv.name}")
    boundary_silu = nearest_upstream_silu(boundary_input)
    if boundary_silu is None:
        raise RuntimeError(f"No upstream SiLU found for {boundary_first_conv.name}")
    sibling_convs = [node for node in fanout_convs(boundary_input) if node is not boundary_first_conv]
    if not sibling_convs:
        raise RuntimeError(f"No sibling fan-out Conv found for {boundary_first_conv.name}")

    return {
        "branch": branch,
        "selected_levels": selected_levels,
        "cls_report": cls_report,
        "cls_silus": [node.name for node in unique_nodes(cls_silus, order)],
        "cls_input_convs": [node.name for node in unique_nodes(cls_input_convs, order)],
        "cls_output_convs": [node.name for node in unique_nodes(cls_output_convs, order)],
        "boundary_silu": [boundary_silu.name],
        "boundary_sibling_convs": [node.name for node in unique_nodes(sibling_convs, order)],
    }


def discover_attention(
    graph_nodes: list[torch.fx.Node],
    order: dict[torch.fx.Node, int],
    expected_attention: int | None,
) -> list[dict[str, str]]:
    attention_groups: dict[str, list[torch.fx.Node]] = defaultdict(list)
    for node in graph_nodes:
        owner = attention_owner(node)
        if owner:
            attention_groups[owner].append(node)
    valid_attention = []
    for owner, nodes in attention_groups.items():
        matmuls = unique_nodes([node for node in nodes if node.target in MATMUL_OPS], order)
        softmaxes = unique_nodes([node for node in nodes if node.target in SOFTMAX_OPS], order)
        qkv_convs = [
            node
            for node in nodes
            if node.target in CONV_OPS and any(path.endswith(".attn.qkv.conv") for path in module_paths(node))
        ]
        pe_convs = [
            node
            for node in nodes
            if node.target in CONV_OPS and any(path.endswith(".attn.pe.conv") for path in module_paths(node))
        ]
        if len(matmuls) != 2 or len(softmaxes) != 1 or len(qkv_convs) != 1 or len(pe_convs) != 1:
            raise RuntimeError(
                f"Incomplete Attention {owner}: matmuls={len(matmuls)}, softmax={len(softmaxes)}, "
                f"qkv_conv={len(qkv_convs)}, pe_conv={len(pe_convs)}"
            )
        scale_muls = [
            node
            for node in nodes
            if node.target in MUL_OPS and any(arg is matmuls[0] for arg in node_args((node.args, node.kwargs)))
        ]
        if len(scale_muls) != 1:
            raise RuntimeError(f"Attention {owner} has {len(scale_muls)} scale Mul node(s), expected 1")
        valid_attention.append(
            {
                "owner": owner,
                "qkv_conv": qkv_convs[0],
                "pe_conv": pe_convs[0],
                "first_matmul": matmuls[0],
                "scale_mul": scale_muls[0],
                "softmax": softmaxes[0],
                "second_matmul": matmuls[1],
            }
        )
    valid_attention.sort(key=lambda item: order[item["first_matmul"]])
    if expected_attention is not None and len(valid_attention) != expected_attention:
        owners = [item["owner"] for item in valid_attention]
        raise RuntimeError(
            f"Found {len(valid_attention)} complete Attention region(s), expected {expected_attention}: {owners}"
        )
    return [
        {key: value.name if isinstance(value, torch.fx.Node) else value for key, value in item.items()}
        for item in valid_attention
    ]


def discover(
    gm: torch.fx.GraphModule,
    branch_arg: str,
    cls_last_n: int,
    expected_attention: int | None,
    enable_cls_u16: bool,
) -> dict[str, Any]:
    graph_nodes = list(gm.graph.nodes)
    order = {node: index for index, node in enumerate(graph_nodes)}
    result: dict[str, Any] = {
        "cls_u16": enable_cls_u16,
        "attention": discover_attention(graph_nodes, order, expected_attention),
    }
    if enable_cls_u16:
        result.update(discover_cls_u16(graph_nodes, order, branch_arg, cls_last_n))
    return result


def qdtype(item: dict[str, Any], key: str) -> str | None:
    return item.get("module_config", {}).get(key, {}).get("dtype")


def select_one(items: list[dict[str, Any]], description: str, predicate) -> dict[str, Any]:
    matches = [item for item in items if predicate(item)]
    if len(matches) != 1:
        raise RuntimeError(f"Template has {len(matches)} {description} entries, expected 1")
    return matches[0]


def is_cls_u16_entry(item: dict[str, Any]) -> bool:
    if item.get("module_names") is None:
        return False
    if item.get("module_type") == "silu":
        return qdtype(item, "input") == "U16" and qdtype(item, "output") == "U16"
    if item.get("module_type") == "conv":
        return qdtype(item, "input") == "U16"
    return False


def template_has_cls_u16(template: dict[str, Any]) -> bool:
    return any(is_cls_u16_entry(item) for item in template.get("regional_configs", []))


def update_config(template: dict[str, Any], result: dict[str, Any], enable_cls_u16: bool) -> dict[str, Any]:
    config = copy.deepcopy(template)
    regional = config.get("regional_configs", [])
    first_matmul_entry = select_one(
        regional,
        "first Attention MatMul S8 output",
        lambda item: (
            item.get("module_type") == "matmul"
            and item.get("module_names") is not None
            and qdtype(item, "output") == "S8"
        ),
    )
    scale_mul_entry = select_one(
        regional,
        "Attention scale Mul S8",
        lambda item: item.get("module_type") == "mul" and qdtype(item, "output") == "S8",
    )
    softmax_entry = select_one(
        regional,
        "Attention Softmax S8",
        lambda item: item.get("module_type") == "softmax" and qdtype(item, "output") == "S8",
    )
    qkv_entry = select_one(
        regional,
        "Attention QKV Conv S8 output",
        lambda item: (
            item.get("module_type") == "conv" and qdtype(item, "input") == "U8" and qdtype(item, "output") == "S8"
        ),
    )
    pe_entry = select_one(
        regional,
        "Attention PE Conv S8 input",
        lambda item: (
            item.get("module_type") == "conv" and qdtype(item, "input") == "S8" and qdtype(item, "output") is None
        ),
    )

    first_matmul_entry["module_names"] = [item["first_matmul"] for item in result["attention"]]
    scale_mul_entry["module_names"] = [item["scale_mul"] for item in result["attention"]]
    softmax_entry["module_names"] = [item["softmax"] for item in result["attention"]]
    qkv_entry["module_names"] = [item["qkv_conv"] for item in result["attention"]]
    pe_entry["module_names"] = [item["pe_conv"] for item in result["attention"]]

    if not enable_cls_u16:
        config["regional_configs"] = [item for item in regional if not is_cls_u16_entry(item)]
        return config

    local_u16_silus = [item for item in regional if item.get("module_type") == "silu" and is_cls_u16_entry(item)]
    cls_silu_entry = select_one(local_u16_silus, "classification SiLU U16", lambda item: len(item["module_names"]) > 1)
    boundary_silu_entry = select_one(local_u16_silus, "boundary SiLU U16", lambda item: len(item["module_names"]) == 1)
    u16_input_convs = [
        item
        for item in regional
        if item.get("module_type") == "conv" and is_cls_u16_entry(item) and qdtype(item, "output") is None
    ]
    cls_conv_entry = select_one(
        u16_input_convs, "classification Conv input U16", lambda item: len(item["module_names"]) > 1
    )
    boundary_conv_entry = select_one(
        u16_input_convs, "boundary fan-out Conv input U16", lambda item: len(item["module_names"]) == 1
    )
    cls_output_entry = select_one(
        regional,
        "classification output Conv U16",
        lambda item: (
            item.get("module_type") == "conv" and qdtype(item, "input") == "U16" and qdtype(item, "output") == "U16"
        ),
    )
    cls_silu_entry["module_names"] = result["cls_silus"]
    boundary_silu_entry["module_names"] = result["boundary_silu"]
    cls_conv_entry["module_names"] = result["cls_input_convs"]
    boundary_conv_entry["module_names"] = result["boundary_sibling_convs"]
    cls_output_entry["module_names"] = result["cls_output_convs"]
    return config


def main() -> None:
    args = parse_args()
    if args.cls_last_n < 1 or (args.expected_attention is not None and args.expected_attention < 1):
        raise ValueError("--cls-last-n and --expected-attention must be positive")
    if not args.check and args.output is None:
        raise ValueError("--output is required unless --check is used")
    if args.output is not None and args.base_config.resolve() == args.output.resolve():
        raise ValueError("--output must not overwrite --base-config")

    template = json.loads(args.base_config.read_text())
    template_cls_u16 = template_has_cls_u16(template)
    enable_cls_u16 = template_cls_u16 if args.cls_u16 == "auto" else args.cls_u16 == "on"
    if enable_cls_u16 and not template_cls_u16:
        raise RuntimeError("--cls-u16 on requires a template containing the clsU16 regional entries")

    model = YOLO(args.model, task="detect")
    if args.pretrained:
        model.load(args.pretrained)
    float_model = model.model.to(args.device).train()
    inputs = torch.rand(args.batch, 3, args.imgsz, args.imgsz, device=args.device)
    exported = torch.export.export_for_training(
        float_model,
        (inputs,),
        dynamic_shapes={"x": {0: Dim.AUTO, 2: Dim.AUTO, 3: Dim.AUTO}},
    )
    gm = exported.module()
    result = discover(gm, args.branch, args.cls_last_n, args.expected_attention, enable_cls_u16)

    generated = update_config(template, result, enable_cls_u16)
    if args.check:
        if generated != template:
            raise SystemExit(
                "ERROR: QAT config node structure does not match the current FX graph; "
                "regenerate it with --output before training or export"
            )
        print("config node structure: PASS")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(generated, indent=4) + "\n")

    report = {
        "model": args.model,
        "pretrained": args.pretrained,
        "base_config": str(args.base_config),
        "output": str(args.output) if args.output else None,
        "cls_u16_mode": args.cls_u16,
        **result,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    print(f"classification U16: {'enabled' if enable_cls_u16 else 'disabled'}")
    if enable_cls_u16:
        print(f"branch: {result['branch']}")
        print(f"classification levels: {result['selected_levels']}")
        print(f"classification SiLU U16: {result['cls_silus']}")
        print(f"classification Conv input U16: {result['cls_input_convs']}")
        print(f"classification Conv input/output U16: {result['cls_output_convs']}")
        print(f"boundary SiLU U16: {result['boundary_silu']}")
        print(f"boundary sibling Conv input U16: {result['boundary_sibling_convs']}")
    for index, item in enumerate(result["attention"], start=1):
        print(f"attention {index}: {item}")
    if args.output:
        print(f"generated config: {args.output}")


if __name__ == "__main__":
    main()
