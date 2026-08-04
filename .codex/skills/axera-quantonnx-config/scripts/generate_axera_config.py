#!/usr/bin/env python3
"""Generate an AXERA Pulsar2 QuantONNX config from QAT Attention regions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import onnx
from onnx import TensorProto

DTYPE_NAMES = {
    TensorProto.INT8: "S8",
    TensorProto.UINT8: "U8",
    TensorProto.INT16: "S16",
    TensorProto.UINT16: "U16",
}
TRANSPARENT_OPS = {"QuantizeLinear", "DequantizeLinear", "Reshape", "Split", "Transpose", "Identity"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path, help="export.py generated *_slim.onnx.")
    parser.add_argument("--output", required=True, type=Path, help="AXERA Pulsar2 JSON config to write.")
    parser.add_argument(
        "--calibration-dataset",
        default="/path/to/dataset",
        help="Required Pulsar2 field; QuantONNX does not consume a calibration dataset.",
    )
    parser.add_argument("--calibration-size", type=int, default=32)
    parser.add_argument("--calibration-format", default="Binary")
    parser.add_argument("--output-dir", default=None, help="Defaults to ./output_<onnx stem>.")
    parser.add_argument("--target-hardware", default="AX650")
    parser.add_argument("--npu-mode", default="NPU3")
    parser.add_argument("--silu-data-type", choices=("U8", "U16"), default="U8")
    return parser.parse_args()


def build_graph(model: onnx.ModelProto):
    producers = {output: node for node in model.graph.node for output in node.output}
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)
    initializers = {item.name: item for item in model.graph.initializer}
    return producers, consumers, initializers


def qdtype(node: onnx.NodeProto | None, initializers: dict[str, onnx.TensorProto]) -> str | None:
    if node is None or node.op_type not in {"QuantizeLinear", "DequantizeLinear"} or len(node.input) < 3:
        return None
    zero_point = initializers.get(node.input[2])
    return DTYPE_NAMES.get(zero_point.data_type) if zero_point is not None else None


def unique(nodes: list[onnx.NodeProto]) -> list[onnx.NodeProto]:
    return list({node.name: node for node in nodes}.values())


def nearest_upstream(start: str, producers, target_ops: set[str], max_depth: int = 20) -> list[onnx.NodeProto]:
    queue = deque([(start, 0)])
    found: list[onnx.NodeProto] = []
    visited: set[str] = set()
    while queue:
        tensor, depth = queue.popleft()
        if tensor in visited or depth > max_depth:
            continue
        visited.add(tensor)
        node = producers.get(tensor)
        if node is None:
            continue
        if node.op_type in target_ops:
            found.append(node)
            continue
        for input_name in node.input:
            queue.append((input_name, depth + 1))
    return unique(found)


def nearest_downstream(start: str, consumers, target_ops: set[str], max_depth: int = 20) -> list[onnx.NodeProto]:
    queue = deque([(start, 0)])
    found: list[onnx.NodeProto] = []
    visited: set[str] = set()
    while queue:
        tensor, depth = queue.popleft()
        if tensor in visited or depth > max_depth:
            continue
        visited.add(tensor)
        for node in consumers.get(tensor, []):
            if node.op_type in target_ops:
                found.append(node)
            elif node.op_type in TRANSPARENT_OPS:
                for output_name in node.output:
                    queue.append((output_name, depth + 1))
    return unique(found)


def exactly_one(nodes: list[onnx.NodeProto], role: str) -> onnx.NodeProto:
    if len(nodes) != 1:
        raise RuntimeError(f"Expected one {role}, found {[node.name for node in nodes]}")
    return nodes[0]


def output_dtypes(node: onnx.NodeProto, consumers, initializers) -> set[str]:
    found: set[str] = set()
    queue = deque(node.output)
    visited: set[str] = set()
    while queue:
        tensor = queue.popleft()
        if tensor in visited:
            continue
        visited.add(tensor)
        for consumer in consumers.get(tensor, []):
            dtype = qdtype(consumer, initializers)
            if dtype:
                found.add(dtype)
            elif consumer.op_type in {"Reshape", "Split", "Transpose", "Identity"}:
                queue.extend(consumer.output)
    return found


def input_dtype(node: onnx.NodeProto, index: int, producers, initializers) -> str | None:
    return qdtype(producers.get(node.input[index]), initializers)


def discover_attention(model: onnx.ModelProto):
    producers, consumers, initializers = build_graph(model)
    regions = []
    for softmax in (node for node in model.graph.node if node.op_type == "Softmax"):
        scale_mul = exactly_one(nearest_upstream(softmax.input[0], producers, {"Mul"}), f"scale Mul for {softmax.name}")
        first_matmul = exactly_one(
            unique(
                [candidate for name in scale_mul.input for candidate in nearest_upstream(name, producers, {"MatMul"})]
            ),
            f"first MatMul for {softmax.name}",
        )
        second_matmul = exactly_one(
            nearest_downstream(softmax.output[0], consumers, {"MatMul"}), f"second MatMul for {softmax.name}"
        )
        qkv_conv = exactly_one(
            unique(
                [candidate for name in first_matmul.input for candidate in nearest_upstream(name, producers, {"Conv"})]
            ),
            f"QKV Conv for {softmax.name}",
        )
        split = exactly_one(
            nearest_downstream(qkv_conv.output[0], consumers, {"Split"}), f"QKV Split for {softmax.name}"
        )
        reshape_before_split = exactly_one(
            nearest_upstream(split.input[0], producers, {"Reshape"}), f"QKV Reshape for {softmax.name}"
        )
        q_transpose = exactly_one(
            unique(
                [
                    candidate
                    for name in first_matmul.input
                    for candidate in nearest_upstream(name, producers, {"Transpose"})
                ]
            ),
            f"Q/K Transpose for {softmax.name}",
        )
        softmax_transpose = exactly_one(
            nearest_downstream(softmax.output[0], consumers, {"Transpose"}), f"Softmax Transpose for {softmax.name}"
        )
        pe_conv = exactly_one(
            nearest_downstream(qkv_conv.output[0], consumers, {"Conv"}), f"PE Conv for {softmax.name}"
        )
        value_reshape = exactly_one(
            nearest_upstream(pe_conv.input[0], producers, {"Reshape"}), f"V Reshape for {softmax.name}"
        )

        core = [
            reshape_before_split,
            split,
            q_transpose,
            first_matmul,
            scale_mul,
            softmax,
            softmax_transpose,
            value_reshape,
        ]
        for node in core:
            if output_dtypes(node, consumers, initializers) != {"S8"}:
                raise RuntimeError(
                    f"{node.name} must have only S8 downstream Q/DQ, got {output_dtypes(node, consumers, initializers)}"
                )
        if output_dtypes(qkv_conv, consumers, initializers) != {"S8"}:
            raise RuntimeError(f"{qkv_conv.name} output must be S8")
        if [input_dtype(first_matmul, index, producers, initializers) for index in range(2)] != ["S8", "S8"]:
            raise RuntimeError(f"{first_matmul.name} inputs must be S8/S8")
        if [input_dtype(second_matmul, index, producers, initializers) for index in range(2)] != ["S8", "S8"]:
            raise RuntimeError(f"{second_matmul.name} inputs must be S8/S8")
        if input_dtype(pe_conv, 0, producers, initializers) != "S8":
            raise RuntimeError(f"{pe_conv.name} activation input must be S8")
        regions.append({"qkv": qkv_conv, "core": core, "second": second_matmul, "pe": pe_conv})
    if not regions:
        raise RuntimeError("No complete Attention region found")
    return regions


def main() -> None:
    args = parse_args()
    model = onnx.load(args.onnx)
    onnx.checker.check_model(model)
    regions = discover_attention(model)

    core = [node.name for region in regions for node in region["core"]]
    qkv = [region["qkv"].name for region in regions]
    s8_inputs = [node.name for region in regions for node in (region["second"], region["pe"])]
    config = {
        "input": str(args.onnx),
        "output_dir": args.output_dir or f"./output_{args.onnx.stem}",
        "model_type": "QuantONNX",
        "target_hardware": args.target_hardware,
        "npu_mode": args.npu_mode,
        "quant": {
            "input_configs": [
                {
                    "tensor_name": "DEFAULT",
                    "calibration_dataset": args.calibration_dataset,
                    "calibration_size": args.calibration_size,
                    "calibration_mean": [0, 0, 0],
                    "calibration_std": [1.0, 1.0, 1.0],
                    "calibration_format": args.calibration_format,
                }
            ],
            "layer_configs": [
                {"op_type": "Silu", "data_type": args.silu_data_type},
                {"layer_names": core, "data_type": "S8", "output_data_type": "S8"},
                {"layer_names": qkv, "output_data_type": "S8"},
                {"layer_names": s8_inputs, "data_type": "S8"},
            ],
            "conv_bias_data_type": "FP32",
            "precision_analysis": True,
            "precision_analysis_method": "PerLayer",
            "precision_analysis_mode": "NPUBackend",
        },
        "input_processors": [
            {
                "tensor_name": "DEFAULT",
                "tensor_layout": "NCHW",
                "src_layout": "NCHW",
                "src_dtype": "FP32",
                "mean": [0, 0, 0],
                "std": [1, 1, 1],
            }
        ],
        "output_processors": [],
        "compiler": {"check": 2},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n")
    print(f"generated {args.output}: Attention={len(regions)}, core={len(core)}, qkv={qkv}, S8 inputs={s8_inputs}")


if __name__ == "__main__":
    main()
