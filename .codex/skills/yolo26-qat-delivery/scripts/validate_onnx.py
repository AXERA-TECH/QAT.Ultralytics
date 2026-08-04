#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

DTYPE_NAMES = {
    TensorProto.INT8: "S8",
    TensorProto.UINT8: "U8",
    TensorProto.INT16: "S16",
    TensorProto.UINT16: "U16",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO26 QAT delivery ONNX model.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--ort", action="store_true", help="Also require ONNX Runtime to load the model.")
    parser.add_argument("--expect-attention-s8", action="store_true")
    parser.add_argument("--expect-bn", type=int, default=0)
    parser.add_argument("--expect-requant", type=int, default=0)
    parser.add_argument("--expect-aligned-split-reshape", type=int, default=2)
    return parser.parse_args()


def qdtype(node, initializers) -> str:
    if node is None or len(node.input) < 3 or node.input[2] not in initializers:
        return "unknown"
    return DTYPE_NAMES.get(initializers[node.input[2]].data_type, str(initializers[node.input[2]].data_type))


def qparams(node, initializers):
    if node is None or len(node.input) < 3:
        return None
    scale = initializers.get(node.input[1])
    zero_point = initializers.get(node.input[2])
    if scale is None or zero_point is None:
        return None
    return numpy_helper.to_array(scale), numpy_helper.to_array(zero_point)


def main() -> None:
    args = parse_args()
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
    requant = [node for node in graph.node if node.op_type == "Identity" and "requant" in node.name]
    errors = []
    if op_counts["BatchNormalization"] != args.expect_bn:
        errors.append(f"BatchNormalization={op_counts['BatchNormalization']}, expected {args.expect_bn}")
    if len(requant) != args.expect_requant:
        errors.append(f"requant Identity={len(requant)}, expected {args.expect_requant}")

    quant_dtypes = Counter(qdtype(node, initializers) for node in graph.node if node.op_type == "QuantizeLinear")
    matmul_domains = []
    for node in graph.node:
        if node.op_type != "MatMul":
            continue
        input_domains = [qdtype(producers.get(name), initializers) for name in node.input]
        output_quantizers = [item for item in consumers.get(node.output[0], []) if item.op_type == "QuantizeLinear"]
        output_domain = qdtype(output_quantizers[0], initializers) if len(output_quantizers) == 1 else "unknown"
        matmul_domains.append((node.name, input_domains, output_domain))

    if args.expect_attention_s8:
        if len(matmul_domains) != 4:
            errors.append(f"MatMul count={len(matmul_domains)}, expected 4")
        if any(domains != ["S8", "S8"] for _, domains, _ in matmul_domains):
            errors.append(f"MatMul input domains are not all S8: {matmul_domains}")
        if Counter(output for _, _, output in matmul_domains) != Counter({"S8": 2, "U8": 2}):
            errors.append(f"MatMul output domains are not S8x2/U8x2: {matmul_domains}")

    aligned_split_reshape = 0
    for split in (node for node in graph.node if node.op_type == "Split"):
        split_params = qparams(producers.get(split.input[0]), initializers)
        if split_params is None:
            continue
        for split_output in split.output:
            for reshape in (node for node in consumers.get(split_output, []) if node.op_type == "Reshape"):
                quantizers = [node for node in consumers.get(reshape.output[0], []) if node.op_type == "QuantizeLinear"]
                if len(quantizers) != 1:
                    continue
                reshape_params = qparams(quantizers[0], initializers)
                if reshape_params is None:
                    continue
                if not all(np.array_equal(left, right) for left, right in zip(split_params, reshape_params)):
                    errors.append(f"Split/Reshape qparams differ: {split.name} -> {reshape.name}")
                else:
                    aligned_split_reshape += 1

    if aligned_split_reshape != args.expect_aligned_split_reshape:
        errors.append(
            f"aligned Split/Reshape branches={aligned_split_reshape}, expected {args.expect_aligned_split_reshape}"
        )

    print(f"model: {args.model}")
    print(f"nodes: {len(graph.node)}")
    print(f"quantize dtypes: {dict(sorted(quant_dtypes.items()))}")
    print(f"BatchNormalization: {op_counts['BatchNormalization']}")
    print(f"requant Identity: {len(requant)}")
    print(f"aligned Split/Reshape branches: {aligned_split_reshape}")
    for name, inputs, output in matmul_domains:
        print(f"MatMul {name}: inputs={inputs}, output={output}")

    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    print("validation: PASS")


if __name__ == "__main__":
    main()
