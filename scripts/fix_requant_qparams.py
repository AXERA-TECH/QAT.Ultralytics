#!/usr/bin/env python3
"""Align Q/DQ parameters around exported ONNX requant marker nodes.

Expected topology::

    QuantizeLinear -> DequantizeLinear -> Identity/requant
        -> QuantizeLinear -> DequantizeLinear

The quantization range with the larger span wins. A private scale and zero-point
initializer is created for every marker so shared initializers outside the
matched Q/DQ chain are not modified accidentally.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}


@dataclass(frozen=True)
class QuantParams:
    """Scalar per-tensor quantization parameters read from a Q/DQ node."""

    node_name: str
    scale_name: str
    zero_point_name: str
    scale: np.ndarray
    zero_point: np.ndarray
    lower: float
    upper: float

    @property
    def span(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class Alignment:
    """Description of one aligned requant marker."""

    marker_name: str
    marker_op_type: str
    selected_from: str
    scale: float
    zero_point: int
    lower: float
    upper: float
    qdq_nodes: tuple[str, ...]


def _is_requant_marker(node: onnx.NodeProto) -> bool:
    op_type = node.op_type.lower()
    return op_type == "identity" or "requant" in op_type or (
        "requant" in node.name.lower() and node.op_type not in QDQ_OPS
    )


def _read_quant_params(
    node: onnx.NodeProto, initializers: dict[str, onnx.TensorProto]
) -> QuantParams | None:
    if node.op_type not in QDQ_OPS or len(node.input) < 3:
        return None
    scale_initializer = initializers.get(node.input[1])
    zero_point_initializer = initializers.get(node.input[2])
    if scale_initializer is None or zero_point_initializer is None:
        return None

    scale = np.asarray(numpy_helper.to_array(scale_initializer))
    zero_point = np.asarray(numpy_helper.to_array(zero_point_initializer))
    if scale.size != 1 or zero_point.size != 1 or zero_point.dtype.kind not in "iu":
        return None

    step = float(scale.reshape(-1)[0])
    if not np.isfinite(step) or step <= 0:
        return None
    zp = int(zero_point.reshape(-1)[0])
    limits = np.iinfo(zero_point.dtype)
    return QuantParams(
        node_name=node.name or node.output[0],
        scale_name=node.input[1],
        zero_point_name=node.input[2],
        scale=scale,
        zero_point=zero_point,
        lower=(int(limits.min) - zp) * step,
        upper=(int(limits.max) - zp) * step,
    )


def _unique_name(base: str, used_names: set[str]) -> str:
    candidate = base
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _marker_qdq_nodes(
    marker: onnx.NodeProto,
    producers: dict[str, onnx.NodeProto],
    consumers: dict[str, list[onnx.NodeProto]],
) -> tuple[list[onnx.NodeProto], str | None]:
    upstream_dqs = [
        producers[name]
        for name in marker.input
        if name in producers and producers[name].op_type == "DequantizeLinear"
    ]
    downstream_qs = [
        node
        for output in marker.output
        for node in consumers.get(output, [])
        if node.op_type == "QuantizeLinear"
    ]
    if len(upstream_dqs) != 1 or len(downstream_qs) != 1:
        return [], f"需要恰好一个上游 DQ 和一个下游 Q，实际为 {len(upstream_dqs)}/{len(downstream_qs)}"

    upstream_dq = upstream_dqs[0]
    upstream_q = producers.get(upstream_dq.input[0]) if upstream_dq.input else None
    if upstream_q is None or upstream_q.op_type != "QuantizeLinear":
        return [], "上游 DQ 不是由 QuantizeLinear 直接产生"

    downstream_q = downstream_qs[0]
    downstream_dqs = [
        node for node in consumers.get(downstream_q.output[0], []) if node.op_type == "DequantizeLinear"
    ]
    if not downstream_dqs:
        return [], "下游 Q 没有对应的 DequantizeLinear 消费者"

    nodes = [upstream_q, upstream_dq, downstream_q, *downstream_dqs]
    unique_nodes = list({id(node): node for node in nodes}.values())
    return unique_nodes, None


def align_requant_qparams(
    model: onnx.ModelProto,
    target_names: set[str] | None = None,
) -> tuple[list[Alignment], list[tuple[str, str]]]:
    """Align scalar Q/DQ parameters around Identity/requant marker nodes.

    Returns the applied alignments and skipped marker reasons. Only exact marker
    names in ``target_names`` are considered when that argument is provided.
    """

    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    producers = {output: node for node in graph.node for output in node.output if output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    used_names = set(initializers)
    used_names.update(value.name for value in graph.input)
    used_names.update(value.name for value in graph.output)
    used_names.update(output for node in graph.node for output in node.output if output)

    alignments: list[Alignment] = []
    skipped: list[tuple[str, str]] = []
    for marker in graph.node:
        if not _is_requant_marker(marker):
            continue
        marker_name = marker.name or marker.output[0]
        if target_names is not None and marker_name not in target_names:
            continue

        qdq_nodes, reason = _marker_qdq_nodes(marker, producers, consumers)
        if reason is not None:
            skipped.append((marker_name, reason))
            continue

        params = [_read_quant_params(node, initializers) for node in qdq_nodes]
        if any(item is None for item in params):
            skipped.append((marker_name, "Q/DQ 参数必须是 initializer 中的有效标量 scale 和 zero-point"))
            continue
        quant_params = [item for item in params if item is not None]
        scale_dtypes = {item.scale.dtype for item in quant_params}
        zero_point_dtypes = {item.zero_point.dtype for item in quant_params}
        if len(scale_dtypes) != 1 or len(zero_point_dtypes) != 1:
            skipped.append((marker_name, "上下游 Q/DQ 的 scale 或 zero-point 数据类型不一致"))
            continue

        # Span is the unambiguous meaning of representable range. The second
        # key only makes equal-span selection deterministic for asymmetric zp.
        selected = max(
            quant_params,
            key=lambda item: (item.span, max(abs(item.lower), abs(item.upper))),
        )
        parameter_inputs = {(node.input[1], node.input[2]) for node in qdq_nodes}
        values_match = all(
            np.array_equal(item.scale, selected.scale)
            and np.array_equal(item.zero_point, selected.zero_point)
            for item in quant_params
        )
        if values_match and len(parameter_inputs) == 1:
            continue

        base = re.sub(r"[^0-9A-Za-z_.-]+", "_", marker_name).strip("_") or "requant"
        scale_name = _unique_name(f"{base}_aligned_scale", used_names)
        zero_point_name = _unique_name(f"{base}_aligned_zero_point", used_names)
        scale_initializer = numpy_helper.from_array(selected.scale.copy(), scale_name)
        zero_point_initializer = numpy_helper.from_array(selected.zero_point.copy(), zero_point_name)
        graph.initializer.extend([scale_initializer, zero_point_initializer])
        initializers[scale_name] = scale_initializer
        initializers[zero_point_name] = zero_point_initializer

        for node in qdq_nodes:
            node.input[1] = scale_name
            node.input[2] = zero_point_name

        alignments.append(
            Alignment(
                marker_name=marker_name,
                marker_op_type=marker.op_type,
                selected_from=selected.node_name,
                scale=float(selected.scale.reshape(-1)[0]),
                zero_point=int(selected.zero_point.reshape(-1)[0]),
                lower=selected.lower,
                upper=selected.upper,
                qdq_nodes=tuple(node.name or node.output[0] for node in qdq_nodes),
            )
        )

    return alignments, skipped


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_requant_aligned{input_path.suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一 ONNX Identity/requant 标记上下游 Q/DQ 的 scale 和 zero-point（大范围优先）"
    )
    parser.add_argument("model", type=Path, help="输入 ONNX 模型")
    parser.add_argument("-o", "--output", type=Path, help="输出路径，默认添加 _requant_aligned 后缀")
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        metavar="NODE_NAME",
        help="只处理指定节点名，可重复使用",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅分析和报告，不保存模型")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.model.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"ONNX 模型不存在: {input_path}")

    output_path = (args.output or _default_output_path(input_path)).resolve()
    if not args.dry_run and output_path == input_path:
        raise ValueError("输出路径不能覆盖输入模型，请指定新的 --output 路径")

    model = onnx.load(input_path, load_external_data=True)
    marker_names = {node.name or node.output[0] for node in model.graph.node if _is_requant_marker(node)}
    requested = set(args.target) if args.target else None
    missing = requested - marker_names if requested else set()
    if missing:
        raise ValueError(f"未找到指定的 Identity/requant 节点: {', '.join(sorted(missing))}")

    print(f"发现 {len(marker_names)} 个 Identity/requant 候选节点")
    alignments, skipped = align_requant_qparams(model, requested)
    for item in alignments:
        print(
            f"[对齐] {item.marker_name} ({item.marker_op_type}): "
            f"采用 {item.selected_from} 的 scale={item.scale:.9g}, zp={item.zero_point}, "
            f"range=[{item.lower:.9g}, {item.upper:.9g}]"
        )
        print(f"       Q/DQ: {', '.join(item.qdq_nodes)}")
    for name, reason in skipped:
        print(f"[跳过] {name}: {reason}")

    if not alignments:
        print("没有需要修改的 requant 参数")
        return 0
    if args.dry_run:
        print(f"dry-run 完成：计划对齐 {len(alignments)} 个位置，未写入文件")
        return 0

    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, output_path)
    print(f"已保存 {output_path}，共对齐 {len(alignments)} 个位置")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
