#!/usr/bin/env python3
"""Validate exported YOLO QAT ONNX structure against its quantization config."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

DTYPE_NAMES = {
    TensorProto.INT8: "S8",
    TensorProto.UINT8: "U8",
    TensorProto.INT16: "S16",
    TensorProto.UINT16: "U16",
}
QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}
DOWNSTREAM_TRANSPARENT_OPS = QDQ_OPS | {"Identity", "Reshape", "Transpose"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Slim Q/DQ ONNX exported by export.py.")
    parser.add_argument("--quant-config", required=True, type=Path)
    parser.add_argument("--ort", action="store_true", help="Also require ONNX Runtime to load the model.")
    parser.add_argument("--expect-bn", type=int, default=0)
    parser.add_argument("--expect-requant", type=int, default=0)
    parser.add_argument(
        "--expect-aligned-split-reshape",
        type=int,
        default=None,
        help="Optionally assert the number of aligned direct Split-to-Reshape branches.",
    )
    parser.add_argument(
        "--skip-output-check",
        action="store_true",
        help="Skip the standard YOLO box/score output dtype contract check.",
    )
    return parser.parse_args()


def config_dtype(item: dict[str, Any], key: str) -> str | None:
    return item.get("module_config", {}).get(key, {}).get("dtype")


def explicit_names(item: dict[str, Any]) -> list[str] | None:
    names = item.get("module_names")
    return names if isinstance(names, list) else None


def matching_entries(config: dict[str, Any], module_type: str, input_dtype=None, output_dtype=None):
    matches = []
    for item in config.get("regional_configs", []):
        if item.get("module_type") != module_type or explicit_names(item) is None:
            continue
        if input_dtype is not None and config_dtype(item, "input") != input_dtype:
            continue
        if output_dtype is not None and config_dtype(item, "output") != output_dtype:
            continue
        matches.append(item)
    return matches


def infer_config_expectations(config: dict[str, Any]) -> tuple[int, int]:
    roles = {
        "first MatMul": matching_entries(config, "matmul", input_dtype="S8", output_dtype="S8"),
        "scale Mul": matching_entries(config, "mul", input_dtype="S8", output_dtype="S8"),
        "Softmax": matching_entries(config, "softmax", input_dtype="S8", output_dtype="S8"),
        "QKV Conv": matching_entries(config, "conv", input_dtype="U8", output_dtype="S8"),
        "PE Conv": [
            item for item in matching_entries(config, "conv", input_dtype="S8") if config_dtype(item, "output") is None
        ],
    }
    counts = {}
    for role, entries in roles.items():
        if len(entries) != 1:
            raise ValueError(f"quant config has {len(entries)} explicit {role} entries, expected 1")
        counts[role] = len(explicit_names(entries[0]) or [])
    if len(set(counts.values())) != 1 or not next(iter(counts.values())):
        raise ValueError(f"Attention regional node counts are inconsistent: {counts}")

    cls_entries = matching_entries(config, "conv", input_dtype="U16", output_dtype="U16")
    if len(cls_entries) > 1:
        raise ValueError(f"quant config has {len(cls_entries)} classification output U16 entries, expected at most 1")
    cls_u16_outputs = len(explicit_names(cls_entries[0]) or []) if cls_entries else 0
    return next(iter(counts.values())), cls_u16_outputs


def constant_tensor(name: str, initializers, producers):
    if name in initializers:
        return initializers[name]
    node = producers.get(name)
    if node is None or node.op_type != "Constant":
        return None
    for attr in node.attribute:
        if attr.name == "value":
            return attr.t
    return None


def qdtype(node, initializers, producers) -> str:
    if node is None or node.op_type not in QDQ_OPS or len(node.input) < 3:
        return "unknown"
    zero_point = constant_tensor(node.input[2], initializers, producers)
    return DTYPE_NAMES.get(zero_point.data_type, str(zero_point.data_type)) if zero_point is not None else "unknown"


def qparams(node, initializers, producers):
    if node is None or node.op_type not in QDQ_OPS or len(node.input) < 3:
        return None
    scale = constant_tensor(node.input[1], initializers, producers)
    zero_point = constant_tensor(node.input[2], initializers, producers)
    if scale is None or zero_point is None:
        return None
    return numpy_helper.to_array(scale), numpy_helper.to_array(zero_point)


def output_qdtype(node, consumers, initializers, producers) -> str:
    quantizers = [consumer for consumer in consumers.get(node.output[0], []) if consumer.op_type == "QuantizeLinear"]
    return qdtype(quantizers[0], initializers, producers) if len(quantizers) == 1 else "unknown"


def nearest_upstream(start_name, producers, target_ops: set[str], max_depth: int = 10):
    queue = deque([(start_name, 0)])
    found = {}
    while queue:
        value, depth = queue.popleft()
        if depth > max_depth:
            continue
        node = producers.get(value)
        if node is None:
            continue
        if node.op_type in target_ops:
            found[node.name] = node
            continue
        for input_name in node.input:
            queue.append((input_name, depth + 1))
    return list(found.values())


def nearest_downstream(start_name, consumers, target_ops: set[str], max_depth: int = 10):
    queue = deque([(start_name, 0)])
    visited = set()
    found = {}
    while queue:
        value, depth = queue.popleft()
        if depth > max_depth:
            continue
        for node in consumers.get(value, []):
            if node.name in visited:
                continue
            visited.add(node.name)
            if node.op_type in target_ops:
                found[node.name] = node
                continue
            if node.op_type in DOWNSTREAM_TRANSPARENT_OPS:
                for output_name in node.output:
                    queue.append((output_name, depth + 1))
    return list(found.values())


def attention_regions(graph, producers, consumers):
    regions = []
    for softmax in (node for node in graph.node if node.op_type == "Softmax"):
        upstream_mul = nearest_upstream(softmax.input[0], producers, {"Mul"})
        if len(upstream_mul) != 1:
            raise ValueError(f"{softmax.name} has {len(upstream_mul)} upstream scale Mul nodes, expected 1")
        first_matmul = []
        for input_name in upstream_mul[0].input:
            first_matmul.extend(nearest_upstream(input_name, producers, {"MatMul"}))
        first_matmul = {node.name: node for node in first_matmul}
        second_matmul = nearest_downstream(softmax.output[0], consumers, {"MatMul"})
        if len(first_matmul) != 1 or len(second_matmul) != 1:
            raise ValueError(
                f"{softmax.name} Attention topology is ambiguous: "
                f"first MatMul={list(first_matmul)}, second MatMul={[node.name for node in second_matmul]}"
            )
        regions.append((next(iter(first_matmul.values())), upstream_mul[0], softmax, second_matmul[0]))
    return regions


def validate_output_contract(graph, cls_u16_outputs, producers, initializers, errors):
    if len(graph.output) % 2:
        errors.append(f"graph outputs={len(graph.output)} cannot be split into equal box/score groups")
        return []

    outputs = [(output.name, qdtype(producers.get(output.name), initializers, producers)) for output in graph.output]
    semantic_outputs = []
    for name, dtype in outputs:
        match = re.fullmatch(r"(boxes|scores)_p(\d+)", name)
        semantic_outputs.append((match.group(1), int(match.group(2)), dtype) if match else None)

    if any(semantic_outputs):
        if not all(semantic_outputs):
            errors.append(f"graph outputs mix semantic and anonymous names: {[name for name, _ in outputs]}")
            return [dtype for _, dtype in outputs]
        boxes = {level: dtype for role, level, dtype in semantic_outputs if role == "boxes"}
        scores = {level: dtype for role, level, dtype in semantic_outputs if role == "scores"}
        if boxes.keys() != scores.keys() or len(boxes) * 2 != len(outputs):
            errors.append(f"box/score output levels differ: boxes={sorted(boxes)}, scores={sorted(scores)}")
            return [dtype for _, dtype in outputs]
        score_dtypes = [scores[level] for level in sorted(scores)]
    else:
        # Backward compatibility for historical exports whose output names were stripped by ONNX slim.
        score_count = len(outputs) // 2
        score_dtypes = [dtype for _, dtype in outputs[score_count:]]

    score_count = len(score_dtypes)
    if cls_u16_outputs > score_count:
        errors.append(f"clsU16 outputs={cls_u16_outputs}, but graph has only {score_count} score outputs")
        return [dtype for _, dtype in outputs]
    expected_scores = ["U8"] * (score_count - cls_u16_outputs) + ["U16"] * cls_u16_outputs
    if score_dtypes != expected_scores:
        errors.append(f"score output dtypes={score_dtypes}, expected {expected_scores}")
    return [dtype for _, dtype in outputs]


def main() -> None:
    args = parse_args()
    config = json.loads(args.quant_config.read_text())
    expected_attention, expected_cls_u16 = infer_config_expectations(config)

    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    if args.ort:
        import onnxruntime as ort

        ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}
    producers = {output: node for node in graph.node for output in node.output}
    consumers = defaultdict(list)
    for node in graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)

    op_counts = Counter(node.op_type for node in graph.node)
    requant_nodes = [node for node in graph.node if "requant" in node.name.lower()]
    errors = []
    if op_counts["BatchNormalization"] != args.expect_bn:
        errors.append(f"BatchNormalization={op_counts['BatchNormalization']}, expected {args.expect_bn}")
    if len(requant_nodes) != args.expect_requant:
        errors.append(f"requant nodes={len(requant_nodes)}, expected {args.expect_requant}")

    regions = attention_regions(graph, producers, consumers)
    if len(regions) != expected_attention:
        errors.append(f"Attention regions={len(regions)}, expected {expected_attention} from config")
    for first_matmul, scale_mul, softmax, second_matmul in regions:
        first_inputs = [qdtype(producers.get(name), initializers, producers) for name in first_matmul.input]
        second_inputs = [qdtype(producers.get(name), initializers, producers) for name in second_matmul.input]
        mul_inputs = [qdtype(producers.get(name), initializers, producers) for name in scale_mul.input]
        first_output = output_qdtype(first_matmul, consumers, initializers, producers)
        mul_output = qdtype(producers.get(softmax.input[0]), initializers, producers)
        softmax_output = output_qdtype(softmax, consumers, initializers, producers)
        second_output = output_qdtype(second_matmul, consumers, initializers, producers)
        if first_inputs != ["S8", "S8"]:
            errors.append(f"{first_matmul.name} input dtypes={first_inputs}, expected S8/S8")
        if first_output != "S8":
            errors.append(f"{first_matmul.name} output dtype={first_output}, expected S8")
        if mul_inputs != ["S8", "S8"]:
            errors.append(f"{scale_mul.name} input dtypes={mul_inputs}, expected S8/S8")
        if mul_output != "S8":
            errors.append(f"{scale_mul.name} output dtype={mul_output}, expected S8")
        if qdtype(producers.get(softmax.input[0]), initializers, producers) != "S8" or softmax_output != "S8":
            errors.append(f"{softmax.name} input/output dtypes={mul_output}/{softmax_output}, expected S8/S8")
        if second_inputs != ["S8", "S8"]:
            errors.append(f"{second_matmul.name} input dtypes={second_inputs}, expected S8/S8")
        if second_output != "U8":
            errors.append(f"{second_matmul.name} output dtype={second_output}, expected U8")

    aligned_split_reshape = 0
    for split in (node for node in graph.node if node.op_type == "Split"):
        split_params = qparams(producers.get(split.input[0]), initializers, producers)
        if split_params is None:
            continue
        for split_output in split.output:
            for reshape in (node for node in consumers.get(split_output, []) if node.op_type == "Reshape"):
                quantizers = [node for node in consumers.get(reshape.output[0], []) if node.op_type == "QuantizeLinear"]
                if len(quantizers) != 1:
                    continue
                reshape_params = qparams(quantizers[0], initializers, producers)
                if reshape_params is None:
                    continue
                if all(np.array_equal(left, right) for left, right in zip(split_params, reshape_params)):
                    aligned_split_reshape += 1
                else:
                    errors.append(f"Split/Reshape qparams differ: {split.name} -> {reshape.name}")
    if args.expect_aligned_split_reshape is not None and aligned_split_reshape != args.expect_aligned_split_reshape:
        errors.append(
            f"aligned Split/Reshape branches={aligned_split_reshape}, expected {args.expect_aligned_split_reshape}"
        )

    output_dtypes = []
    if not args.skip_output_check:
        output_dtypes = validate_output_contract(graph, expected_cls_u16, producers, initializers, errors)

    quant_dtypes = Counter(
        qdtype(node, initializers, producers) for node in graph.node if node.op_type == "QuantizeLinear"
    )
    print(f"model: {args.model}")
    print(f"quant config: {args.quant_config}")
    print(f"nodes: {len(graph.node)}")
    print(f"quantize dtypes: {dict(sorted(quant_dtypes.items()))}")
    print(f"Attention regions: {len(regions)}")
    print(f"classification U16 outputs: {expected_cls_u16}")
    if output_dtypes:
        print(f"graph output dtypes: {output_dtypes}")
    print(f"BatchNormalization: {op_counts['BatchNormalization']}")
    print(f"requant nodes: {len(requant_nodes)}")
    print(f"aligned Split/Reshape branches: {aligned_split_reshape}")
    for index, (first_matmul, scale_mul, softmax, second_matmul) in enumerate(regions, start=1):
        print(f"Attention {index}: {first_matmul.name} -> {scale_mul.name} -> {softmax.name} -> {second_matmul.name}")

    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    print("structure validation: PASS")


if __name__ == "__main__":
    main()
