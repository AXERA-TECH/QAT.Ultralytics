from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
from onnxslim import slim
from torch.ao.quantization.quantize_pt2e import convert_pt2e

from ultralytics import YOLO
from ultralytics.utils.qat_utils import prepare_pt2e_qat_model, resolve_qat_config_path

warnings.filterwarnings("ignore", message=r"erase_node\(batch_norm_.*")


@dataclass(frozen=True)
class ExportDefaults:
    task: str
    model: str
    pretrained: str
    qat_weights: str
    out: str
    qat_state_out: str


DEFAULT_EXPORTS = {
    "detect": ExportDefaults(
        task="detect",
        model="yolo26n.yaml",
        pretrained="yolo26n.pt",
        qat_weights="runs/detect/exp32-yolo26n-S16matmul-e2eFalse/weights/best.pt",
        out="./yolo26_onnx/qat_exp32_one2many.onnx",
        qat_state_out="./yolo26_onnx/qat_exp32_one2many.pth",
    ),
    "segment": ExportDefaults(
        task="segment",
        model="yolo26n-seg.yaml",
        pretrained="./weights/yolo26n-seg.pt",
        qat_weights="runs/segment/qat2/weights/best.pt",
        out="./qat-seg.onnx",
        qat_state_out="./qat-seg.pth",
    ),
    "obb": ExportDefaults(
        task="obb",
        model="yolo26n-obb.yaml",
        pretrained="./weights/yolo26n-obb.pt",
        qat_weights="runs/obb/yolo26n-obb-qat/weights/best.pt",
        out="./yolo26_onnx/yolo26n_obb_qat.onnx",
        qat_state_out="./yolo26_onnx/yolo26n_obb_qat.pth",
    ),
    "pose": ExportDefaults(
        task="pose",
        model="yolo26n-pose.yaml",
        pretrained="./weights/yolo26n-pose.pt",
        qat_weights="runs/pose/yolo26n-pose-qat/weights/best.pt",
        out="./yolo26_onnx/yolo26n_pose_qat.onnx",
        qat_state_out="./yolo26_onnx/yolo26n_pose_qat.pth",
    ),
    "classify": ExportDefaults(
        task="classify",
        model="yolo26n-cls.yaml",
        pretrained="./weights/yolo26n-cls.pt",
        qat_weights="runs/classify/yolo26n-cls-qat/weights/best.pt",
        out="./yolo26_onnx/yolo26n_cls_qat.onnx",
        qat_state_out="./yolo26_onnx/yolo26n_cls_qat.pth",
    ),
}

REQUIRED_ONNX_OPSET = 21  # int16 Q/DQ 需要更高 opset，ORT 在 opset 18 下会拒绝加载


def parse_bool(value: str | bool) -> bool:
    """Parse a command-line boolean value."""
    if isinstance(value, bool):
        return value
    return value.lower() in {"true", "1", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO26 PT2E QAT model to ONNX.")
    parser.add_argument("--task", choices=sorted(DEFAULT_EXPORTS), default="detect")
    parser.add_argument("--model", default=None, help="Model yaml path.")
    parser.add_argument("--pretrained", default="yolo26n.pt", help="Float pretrained weights path.")
    parser.add_argument(
        "--qat-weights",
        dest="qat_weights",
        default="runs/detect/exp58-globalSiluU8AttnS8-e2eTrue-noEMA/weights/best.pt",
        help="QAT checkpoint path.",
    )
    parser.add_argument("--out", default="./yolo26_onnx/exp58_one2one.onnx", help="ONNX output path.")
    parser.add_argument(
        "--qat-state-out",
        dest="qat_state_out",
        default="./yolo26_onnx/qat_exp32_one2many.pth",
        help="Saved QAT state dict path.",
    )
    parser.add_argument(
        "--export-pth",
        dest="export_pth",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Export the loaded QAT state dict to --qat-state-out (default: false).",
    )
    parser.add_argument(
        "--quant-config",
        dest="quant_config",
        default="config-qat/config_siluInU8_attnS8_clsU16.json",
        help="Quant config path.",
    )
    parser.add_argument("--device", default="cuda", help="Export device.")
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[640, 640],
        help="Input image size, e.g. --imgsz 640 640 or --imgsz 640",
    )
    parser.add_argument(
        "--end2end",
        dest="end2end",
        default=True,
        type=parse_bool,
        help="Export one2one branch (True) or one2many branch (False) for NMS.",
    )
    parser.add_argument(
        "--fix-split-reshape-quant",
        dest="fix_split_reshape_quant",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Align covered Reshape Q/DQ ranges to Split input quantization for AXERA NPU (default: true).",
    )
    return parser.parse_args()


class DetectOne2OneWrapper(torch.nn.Module):
    """Export only one-to-one branch outputs for QAT detection inference (per-scale, no concat)."""

    def __init__(self, qat_model: torch.nn.Module):
        super().__init__()
        self.qat_model = qat_model

    def forward(self, x: torch.Tensor):
        preds = self.qat_model(x)
        o2o = preds.get("one2one", preds)
        if isinstance(o2o, list):
            return (
                o2o[0]["boxes"],
                o2o[0]["scores"],
                o2o[1]["boxes"],
                o2o[1]["scores"],
                o2o[2]["boxes"],
                o2o[2]["scores"],
            )
        boxes, scores = o2o["boxes"], o2o["scores"]
        if isinstance(boxes, (list, tuple)) and isinstance(scores, (list, tuple)):
            return boxes[0], scores[0], boxes[1], scores[1], boxes[2], scores[2]
        return boxes, scores


class DetectOne2ManyWrapper(torch.nn.Module):
    """Export one-to-many branch outputs for QAT detection inference (per-scale, no concat, no feats)."""

    def __init__(self, qat_model: torch.nn.Module):
        super().__init__()
        self.qat_model = qat_model

    def forward(self, x: torch.Tensor):
        preds = self.qat_model(x)
        o2m = preds.get("one2many", preds)
        if isinstance(o2m, list):
            return (
                o2m[0]["boxes"],
                o2m[0]["scores"],
                o2m[1]["boxes"],
                o2m[1]["scores"],
                o2m[2]["boxes"],
                o2m[2]["scores"],
            )
        boxes, scores = o2m["boxes"], o2m["scores"]
        if isinstance(boxes, (list, tuple)) and isinstance(scores, (list, tuple)):
            return boxes[0], scores[0], boxes[1], scores[1], boxes[2], scores[2]
        return boxes, scores


class SegmentOne2OneWrapper(torch.nn.Module):
    """Export flattened one-to-one predictions and proto tensors for segmentation inference."""

    def __init__(self, qat_model: torch.nn.Module):
        super().__init__()
        self.qat_model = qat_model

    def forward(self, x: torch.Tensor):
        preds = self.qat_model(x)["one2one"]
        boxes = tuple(preds["boxes"]) if isinstance(preds["boxes"], list) else (preds["boxes"],)
        scores = tuple(preds["scores"]) if isinstance(preds["scores"], list) else (preds["scores"],)
        proto = preds["proto"]
        proto_outputs = tuple(proto) if isinstance(proto, (tuple, list)) else (proto,)
        return (
            *boxes,
            *scores,
            preds["mask_coefficient"],
            *proto_outputs,
        )


class OBBWrapper(torch.nn.Module):
    """Export the OBB loss-compatible concatenated box, class, and angle tensors."""

    def __init__(self, qat_model: torch.nn.Module, end2end: bool):
        super().__init__()
        self.qat_model = qat_model
        self.end2end = end2end

    def forward(self, x: torch.Tensor):
        preds = self.qat_model(x)
        branch = preds["one2one"] if self.end2end else preds["one2many"]
        return branch["boxes"], branch["scores"], branch["angle"]


class PoseWrapper(torch.nn.Module):
    """Export concatenated pose box, class, and keypoint tensors."""

    def __init__(self, qat_model: torch.nn.Module, end2end: bool):
        super().__init__()
        self.qat_model = qat_model
        self.end2end = end2end

    def forward(self, x: torch.Tensor):
        preds = self.qat_model(x)
        branch = preds["one2one"] if self.end2end else preds["one2many"]
        return branch["boxes"], branch["scores"], branch["kpts"]


class ClassifyWrapper(torch.nn.Module):
    """Export raw classification logits from the single-logits QAT graph."""

    def __init__(self, qat_model: torch.nn.Module):
        super().__init__()
        self.qat_model = qat_model

    def forward(self, x: torch.Tensor):
        # The train-traced Classify graph outputs raw logits, fully quantized (ends in Q/DQ).
        # Do NOT append softmax here: it is not part of the QAT graph, so a float tail op would
        # leave the graph output un-quantized. argmax(logits) == argmax(softmax); host applies softmax.
        return self.qat_model(x)


def resolve_paths(args: argparse.Namespace) -> ExportDefaults:
    defaults = DEFAULT_EXPORTS[args.task]
    return ExportDefaults(
        task=args.task,
        model=args.model or defaults.model,
        pretrained=args.pretrained or defaults.pretrained,
        qat_weights=args.qat_weights or defaults.qat_weights,
        out=args.out or defaults.out,
        qat_state_out=args.qat_state_out or defaults.qat_state_out,
    )


def normalize_imgsz(imgsz: list[int]) -> list[int]:
    if len(imgsz) == 1:
        h = w = int(imgsz[0])
    elif len(imgsz) == 2:
        h, w = (int(v) for v in imgsz)
    else:
        raise ValueError(f"Unsupported imgsz {imgsz}. Use one or two integers.")
    return [1, 3, h, w]


def remove_invalid_graph_outputs(model: onnx.ModelProto) -> list[str]:
    """Remove graph outputs that are no longer produced after slim passes."""
    producer_outputs = {output for node in model.graph.node for output in node.output if output}
    input_names = {value.name for value in model.graph.input}
    initializer_names = {value.name for value in model.graph.initializer}
    valid_output_names = producer_outputs | input_names | initializer_names

    removed = [output.name for output in model.graph.output if output.name not in valid_output_names]
    if removed:
        kept_outputs = [output for output in model.graph.output if output.name in valid_output_names]
        del model.graph.output[:]
        model.graph.output.extend(kept_outputs)
    return removed


def ensure_main_opset(proto, version: int) -> None:
    """Bump the main ONNX opset for model and local functions when exporter emits int16 Q/DQ."""
    found = False
    for opset in proto.opset_import:
        if opset.domain == "":
            opset.version = max(opset.version, version)
            found = True
    if not found:
        proto.opset_import.add(domain="", version=version)

    functions = getattr(proto, "functions", None)
    if functions:
        for function in functions:
            ensure_main_opset(function, version)


def build_quantized_model(
    cfg: ExportDefaults,
    quant_config: str,
    device: str,
    qat_onnx_imgsz: list[int],
    export_pth: bool = False,
):
    model = YOLO(cfg.model, task=cfg.task)
    if cfg.task in {"obb", "pose"}:
        # Task checkpoints may use dataset-specific heads while architecture YAML defaults to COCO.
        # Rebuild the head with the checkpoint metadata before preparing the PT2E graph.
        source_model = YOLO(cfg.pretrained, task=cfg.task)
        kwargs = {"nc": len(source_model.names), "ch": 3}
        if cfg.task == "pose":
            kwargs["data_kpt_shape"] = source_model.model.model[-1].kpt_shape
        model.model = model.task_map[cfg.task]["model"](cfg.model, **kwargs)
    elif cfg.task == "classify" and cfg.qat_weights and Path(cfg.qat_weights).exists():
        # Classification fine-tunes the head to the training dataset's class count; recover it from
        # the QAT checkpoint's train_args so the prepared graph matches the saved observers.
        from ultralytics.data.utils import check_cls_dataset

        cls_data = torch.load(cfg.qat_weights, weights_only=False).get("train_args", {}).get("data")
        if cls_data:
            model.model = model.task_map["classify"]["model"](cfg.model, nc=check_cls_dataset(cls_data)["nc"], ch=3)
    model.load(cfg.pretrained)
    weight_dict = None
    train_args = {}

    # Load checkpoint metadata to detect training config
    if cfg.qat_weights and Path(cfg.qat_weights).exists():
        weight_dict = torch.load(cfg.qat_weights, weights_only=False)
        train_args = weight_dict.get("train_args", {})
        ckpt_quant_config = (
            train_args.get("qat_config", None)
            if isinstance(train_args, dict)
            else getattr(train_args, "qat_config", None)
        )
        if ckpt_quant_config and resolve_qat_config_path(ckpt_quant_config).is_file():
            quant_config = str(resolve_qat_config_path(ckpt_quant_config))
            print(f"Using quant config from checkpoint: {quant_config}")

    quant_config = str(resolve_qat_config_path(quant_config))

    if isinstance(train_args, dict):
        dynamic_batch_max = int(train_args.get("qat_dynamic_batch_max", 128))
        use_lsq = bool(train_args.get("qat_lsq", False))
    else:
        dynamic_batch_max = int(getattr(train_args, "qat_dynamic_batch_max", 128))
        use_lsq = bool(getattr(train_args, "qat_lsq", False))
    float_model = model.model.to(device).train()
    height, width = qat_onnx_imgsz[-2:]
    inputs = torch.rand(*qat_onnx_imgsz, device=device)
    print("start export!")
    _, prepared_model = prepare_pt2e_qat_model(
        float_model=float_model,
        device=device,
        config_path=quant_config,
        imgsz=(height, width),
        dynamic_batch_max=dynamic_batch_max,
        use_lsq=use_lsq,
    )
    print("prepared training graph done!")

    if weight_dict is not None:
        qat_weight_dict = weight_dict.get("qat_ema") or weight_dict.get("qat_model")
        if qat_weight_dict is None:
            raise KeyError("Checkpoint missing 'qat_ema' and 'qat_model' keys")
        if export_pth:
            torch.save(qat_weight_dict, cfg.qat_state_out)
            print(f"export QAT state dict to [{cfg.qat_state_out}] done!")
        prepared_model.load_state_dict(qat_weight_dict)
        print("load_state_dict done!")
    else:
        print("skip weight loading (no checkpoint); initialize observers with one random batch")

    # Loaded observer buffers may retain the checkpoint device. Uninitialized observers also
    # need one target-device forward so calculate_qparams does not create defaults on CUDA0.
    prepared_model.to(device)
    prepared_model.eval()
    if weight_dict is None:
        with torch.no_grad():
            prepared_model(inputs)

    quantized_model = convert_pt2e(prepared_model)
    print("convert_pt2e done!")
    return quantized_model, inputs


def get_segment_proto_names(proto) -> list[str]:
    if isinstance(proto, (tuple, list)):
        if len(proto) == 2:
            return ["proto_masks", "proto_semseg"]
        return [f"proto_{i}" for i in range(len(proto))]
    return ["proto_masks"]


def build_export_plan(task: str, quantized_model: torch.nn.Module, inputs: torch.Tensor, end2end: bool = True):
    sample_outputs = quantized_model(inputs)
    if task == "detect":
        if end2end:
            return DetectOne2OneWrapper(quantized_model), [
                "boxes_p3",
                "scores_p3",
                "boxes_p4",
                "scores_p4",
                "boxes_p5",
                "scores_p5",
            ]
        else:
            return DetectOne2ManyWrapper(quantized_model), [
                "boxes_p3",
                "scores_p3",
                "boxes_p4",
                "scores_p4",
                "boxes_p5",
                "scores_p5",
            ]
    if task == "segment":
        proto_names = get_segment_proto_names(sample_outputs["one2one"]["proto"])
        output_names = [
            "boxes_p3",
            "boxes_p4",
            "boxes_p5",
            "scores_p3",
            "scores_p4",
            "scores_p5",
            "mask_coefficient",
            *proto_names,
        ]
        return SegmentOne2OneWrapper(quantized_model), output_names
    if task == "obb":
        branch = sample_outputs["one2one"] if end2end else sample_outputs["one2many"]
        if not all(isinstance(branch.get(key), torch.Tensor) for key in ("boxes", "scores", "angle")):
            raise RuntimeError("OBB export requires concatenated box, score, and angle tensors")
        return OBBWrapper(quantized_model, end2end=end2end), [
            "boxes",
            "scores",
            "angle",
        ]
    if task == "pose":
        branch = sample_outputs["one2one"] if end2end else sample_outputs["one2many"]
        if not all(isinstance(branch.get(key), torch.Tensor) for key in ("boxes", "scores", "kpts")):
            raise RuntimeError("Pose export requires concatenated box, score, and keypoint tensors")
        return PoseWrapper(quantized_model, end2end=end2end), [
            "boxes",
            "scores",
            "keypoints",
        ]
    if task == "classify":
        if not isinstance(sample_outputs, torch.Tensor) or sample_outputs.ndim != 2:
            raise RuntimeError("Classify export requires a single (batch, nc) logits tensor")
        return ClassifyWrapper(quantized_model), ["logits"]
    raise ValueError(f"Unsupported task {task}")


def _fix_qdq_qdq_mismatch(model: onnx.ModelProto, threshold: float = 1.02) -> int:
    """Insert Identity (requant) nodes between adjacent Q-DQ pairs with mismatched quantization.

    In a Q→DQ→Q chain, if the first Q's scale and the second Q's scale differ beyond ``threshold`` (or their zero_points
    differ), deployment tools cannot fuse them into a single requant operation. Inserting an explicit Identity node
    severs the adjacency and marks the position where a **requant** (dequantize→requantize) is required.
    """
    import numpy as np
    from onnx import helper

    init_dict = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    insertions: list[tuple] = []

    for dq_node in model.graph.node:
        if dq_node.op_type != "DequantizeLinear":
            continue
        q_first = next(
            (n for n in model.graph.node if n.op_type == "QuantizeLinear" and n.output[0] == dq_node.input[0]), None
        )
        q_second = next(
            (n for n in model.graph.node if n.op_type == "QuantizeLinear" and dq_node.output[0] in n.input), None
        )
        if q_first is None or q_second is None:
            continue

        s1 = float(init_dict.get(q_first.input[1], np.array([0])).flatten()[0])
        s2 = float(init_dict.get(q_second.input[1], np.array([0])).flatten()[0])
        if s1 < 1e-10 or s2 < 1e-10:
            continue

        zp1 = int(init_dict.get(q_first.input[2], np.array([0])).flatten()[0])
        zp2 = int(init_dict.get(q_second.input[2], np.array([0])).flatten()[0])
        scale_ratio = max(s1, s2) / min(s1, s2)
        zp_mismatch = zp1 != zp2

        if scale_ratio < threshold and not zp_mismatch:
            continue

        name = dq_node.name.replace("/", "_").replace(".", "_")
        identity_output = f"{name}_requant_id"
        identity_node = helper.make_node(
            "Identity",
            inputs=[dq_node.output[0]],
            outputs=[identity_output],
            name=f"{name}_requant",
        )
        reason = f"scale_ratio={scale_ratio:.2f}"
        if zp_mismatch:
            reason += f", zp_diff={zp1}!={zp2}"
        print(f"  requant @ {dq_node.name}: {reason}  (s1={s1:.6f}, s2={s2:.6f})")
        insertions.append((identity_node, q_second, dq_node.output[0], identity_output))

    if not insertions:
        return 0

    nodes = list(model.graph.node)
    for id_node, q_second, old_out, new_out in insertions:
        q2_idx = nodes.index(q_second)
        nodes.insert(q2_idx, id_node)
        for i, inp in enumerate(q_second.input):
            if inp == old_out:
                q_second.input[i] = new_out
                break

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return len(insertions)


def _merge_adjacent_dq_q(model: onnx.ModelProto, threshold: float = 1.02) -> int:
    """Merge adjacent DQ→Q pairs with nearly identical scales into a single pass-through.

    PT2E may insert redundant DQ→Q pairs between adjacent layers when their observer scales differ slightly (< 2%).
    These pairs are a no-op passthrough that can be eliminated.
    """
    import numpy as np

    init_dict = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    nodes_to_remove = set()
    merged = 0

    for dq_node in list(model.graph.node):
        if dq_node.op_type != "DequantizeLinear":
            continue

        # Find Q that directly consumes this DQ output
        q_next = next(
            (n for n in model.graph.node if n.op_type == "QuantizeLinear" and dq_node.output[0] == n.input[0]), None
        )
        if q_next is None:
            continue

        # Ensure DQ has exactly one consumer (the Q_next)
        dq_consumers = [n for n in model.graph.node if dq_node.output[0] in n.input]
        if len(dq_consumers) != 1:
            continue

        # Find Q that feeds this DQ
        q_prev = next(
            (n for n in model.graph.node if n.op_type == "QuantizeLinear" and n.output[0] == dq_node.input[0]), None
        )
        if q_prev is None:
            continue

        # Find DQ that Q_next feeds, ensure Q_next has single consumer
        q_next_consumers = [n for n in model.graph.node if q_next.output[0] in n.input]
        if len(q_next_consumers) != 1:
            continue
        dq_next = q_next_consumers[0]
        if dq_next.op_type != "DequantizeLinear":
            continue

        # Check scale proximity
        s_dq = float(init_dict.get(dq_node.input[1], np.array([0])).flatten()[0])
        s_q = float(init_dict.get(q_next.input[1], np.array([0])).flatten()[0])
        if s_dq < 1e-10 or s_q < 1e-10:
            continue
        if max(s_dq, s_q) / min(s_dq, s_q) >= threshold:
            continue

        # Check zero_point consistency (affine quantization may have non-zero zp)
        zp_dq = int(init_dict.get(dq_node.input[2], np.array([0])).flatten()[0])
        zp_q = int(init_dict.get(q_next.input[2], np.array([0])).flatten()[0])
        if zp_dq != zp_q:
            continue

        # Merge: skip DQ+Q_next, rewire Q_prev output to DQ_next input
        dq_next.input[0] = q_prev.output[0]
        nodes_to_remove.add(dq_node.name)
        nodes_to_remove.add(q_next.name)
        merged += 1

    if nodes_to_remove:
        new_nodes = [n for n in model.graph.node if n.name not in nodes_to_remove]
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)

    # Second pass: handle DQ with multiple consumers where one consumer is a redundant Q->DQ branch
    for dq_node in list(model.graph.node):
        if dq_node.op_type != "DequantizeLinear":
            continue

        dq_consumers = [n for n in model.graph.node if dq_node.output[0] in n.input]
        s_dq = float(init_dict.get(dq_node.input[1], np.array([0])).flatten()[0])
        if s_dq < 1e-10:
            continue

        for consumer in dq_consumers:
            if consumer.op_type != "QuantizeLinear":
                continue

            s_q = float(init_dict.get(consumer.input[1], np.array([0])).flatten()[0])
            if s_q < 1e-10:
                continue
            if max(s_dq, s_q) / min(s_dq, s_q) >= threshold:
                continue

            # Check zero_point consistency
            zp_dq = int(init_dict.get(dq_node.input[2], np.array([0])).flatten()[0])
            zp_q = int(init_dict.get(consumer.input[2], np.array([0])).flatten()[0])
            if zp_dq != zp_q:
                continue

            # Find the DQ that this intermediate Q feeds
            q_consumers = [n for n in model.graph.node if consumer.output[0] in n.input]
            if len(q_consumers) != 1:
                continue
            dq_mid = q_consumers[0]
            if dq_mid.op_type != "DequantizeLinear":
                continue

            # Ensure dq_mid has at least one consumer that would benefit from bypass
            dq_mid_consumers = [n for n in model.graph.node if dq_mid.output[0] in n.input]

            # Rewire dq_mid consumers to use parent DQ output directly
            for ds_consumer in dq_mid_consumers:
                for i, inp in enumerate(ds_consumer.input):
                    if inp == dq_mid.output[0]:
                        ds_consumer.input[i] = dq_node.output[0]
                        break

            nodes_to_remove.add(consumer.name)
            nodes_to_remove.add(dq_mid.name)
            merged += 1

    if nodes_to_remove:
        new_nodes = [n for n in model.graph.node if n.name not in nodes_to_remove]
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)

    return merged


def _split_shared_mul_dequant(model: onnx.ModelProto) -> int:
    """Split DequantizeLinear nodes that feed into multiple Mul ops.

    When a single DQ output is consumed by two or more Mul nodes, deployment tools may fail to relate the separate
    DQ→Mul→Q chains back to a shared quantized tensor. Creating per-Mul copies of the DQ resolves this.
    """
    from onnx import helper

    {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    new_nodes = []
    split_count = 0

    for dq_node in model.graph.node:
        if dq_node.op_type != "DequantizeLinear":
            continue
        mul_consumers = [n for n in model.graph.node if n.op_type == "Mul" and dq_node.output[0] in n.input]
        if len(mul_consumers) <= 1:
            continue

        dq_out = dq_node.output[0]
        scale_name = dq_node.input[1]
        zp_name = dq_node.input[2] if len(dq_node.input) > 2 else None

        for mul_node in mul_consumers[1:]:
            new_output = f"{dq_node.output[0]}_split_{mul_node.name.split('/')[-1]}"
            new_dq = helper.make_node(
                "DequantizeLinear",
                inputs=[dq_node.input[0], scale_name] + ([zp_name] if zp_name else []),
                outputs=[new_output],
                name=f"{dq_node.name}_split_{split_count}",
            )
            new_nodes.append((new_dq, mul_node))
            for i, inp in enumerate(mul_node.input):
                if inp == dq_out:
                    mul_node.input[i] = new_output
                    break
            split_count += 1

    if new_nodes:
        nodes = list(model.graph.node)
        for new_dq, mul_node in new_nodes:
            idx = nodes.index(mul_node)
            nodes.insert(idx, new_dq)  # insert before its Mul consumer
        del model.graph.node[:]
        model.graph.node.extend(nodes)

    return split_count


def _merge_near_equal_requant(model: onnx.ModelProto, scale_threshold: float = 1.02, zp_tol: int = 1) -> int:
    """严格合并 onnx 导出/slim 把量化图里【同一 observer】拆成的近乎等价冗余重量化。.

    量化图(prepare/convert_pt2e)里 silu output 与下游共享同一 observer、零 requant； torch.onnx.export(dynamo) 分解 silu + slim 会把它拆成
    scale 微异的 Q/DQ，产生冗余重量化。 结构：q_prev → dq_node → q_next → dq_next，其中 dq_node 可能 fan-out（silu 输出喂多个分支）。
    严格条件(全满足才合并)：scale 比 < scale_threshold、|zp 差| <= zp_tol、四个量化点 dtype 全一致、 q_next 单消费者且为 DQ。合并：把 dq_next 的消费者改回直接用
    dq_node 输出、删 q_next+dq_next （保留 dq_node 供其他 fan-out 分支），等价于把该分支统一到上游 scale/zp、零精度损失。
    """
    import numpy as np

    init_dict = {i.name: onnx.numpy_helper.to_array(i) for i in model.graph.initializer}
    zp_dt = {i.name: i.data_type for i in model.graph.initializer}
    removals: set = set()
    merged = 0

    for dq_node in list(model.graph.node):
        if dq_node.op_type != "DequantizeLinear":
            continue
        q_prev = next(
            (n for n in model.graph.node if n.op_type == "QuantizeLinear" and n.output[0] == dq_node.input[0]), None
        )
        if q_prev is None:
            continue
        s_dq = float(init_dict.get(dq_node.input[1], np.array([0])).flatten()[0])
        if s_dq < 1e-10:
            continue
        zp_dq = int(init_dict.get(dq_node.input[2], np.array([0])).flatten()[0])

        # dq_node 的每个 Q 消费者（fan-out：其中一些是冗余重量化分支）
        for q_next in [n for n in model.graph.node if n.op_type == "QuantizeLinear" and dq_node.output[0] in n.input]:
            if q_next.name in removals:
                continue
            # 严格：q_next 单消费者且为 DQ（只处理 Q→DQ 这对重量化）
            q_next_cons = [n for n in model.graph.node if q_next.output[0] in n.input]
            if len(q_next_cons) != 1 or q_next_cons[0].op_type != "DequantizeLinear":
                continue
            dq_next = q_next_cons[0]

            s_q = float(init_dict.get(q_next.input[1], np.array([0])).flatten()[0])
            if s_q < 1e-10:
                continue
            if max(s_dq, s_q) / min(s_dq, s_q) >= scale_threshold:
                continue
            zp_q = int(init_dict.get(q_next.input[2], np.array([0])).flatten()[0])
            if abs(zp_dq - zp_q) > zp_tol:
                continue
            # dtype 严格一致（四个量化点同位宽）
            dts = {zp_dt.get(x) for x in (q_prev.input[2], dq_node.input[2], q_next.input[2], dq_next.input[2])}
            if len(dts) != 1 or None in dts:
                continue

            # 合并：dq_next 的所有消费者改用 dq_node.output（保留 dq_node，fan-out 安全）；删 q_next + dq_next
            for c in model.graph.node:
                for i, inp in enumerate(c.input):
                    if inp == dq_next.output[0]:
                        c.input[i] = dq_node.output[0]
            removals.add(q_next.name)
            removals.add(dq_next.name)
            merged += 1

    if removals:
        new_nodes = [n for n in model.graph.node if n.name not in removals]
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
    return merged


def _align_split_reshape_quantization(model: onnx.ModelProto) -> int:
    """Align safe Split fan-out Reshape Q/DQ ranges for AXERA backends.

    AXERA NPU treats Split as a passive operator and forces its input/output quantization parameters to match. If one
    Split output also feeds a Reshape branch with a smaller Q/DQ range, the backend may incorrectly select that smaller
    range for the whole Split. Replace only same-dtype scalar Reshape Q/DQ parameters whose range is fully covered by
    the Split input range. New initializers keep unrelated users of shared parameters intact.
    """
    import numpy as np

    graph = model.graph
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    producers = {output: node for node in graph.node for output in node.output if output}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    used_names = set(initializers) | {value.name for value in graph.input} | {value.name for value in graph.output}
    used_names.update(output for node in graph.node for output in node.output if output)

    def quant_params(node):
        if node is None or node.op_type not in {"QuantizeLinear", "DequantizeLinear"} or len(node.input) < 3:
            return None
        scale_init = initializers.get(node.input[1])
        zp_init = initializers.get(node.input[2])
        if scale_init is None or zp_init is None:
            return None
        scale = onnx.numpy_helper.to_array(scale_init)
        zero_point = onnx.numpy_helper.to_array(zp_init)
        if scale.size != 1 or zero_point.size != 1 or zero_point.dtype.kind not in "iu":
            return None
        return scale, zero_point

    def quant_range(scale, zero_point):
        limits = np.iinfo(zero_point.dtype)
        step = float(scale.reshape(-1)[0])
        zp = int(zero_point.reshape(-1)[0])
        return (limits.min - zp) * step, (limits.max - zp) * step

    def unique_name(base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in used_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        return candidate

    aligned = 0
    for split in [node for node in graph.node if node.op_type == "Split" and node.input]:
        split_input_qdq = producers.get(split.input[0])
        split_params = quant_params(split_input_qdq)
        if split_params is None:
            continue
        split_scale, split_zp = split_params
        split_lo, split_hi = quant_range(split_scale, split_zp)

        for split_output in split.output:
            branch_consumers = consumers.get(split_output, [])
            if len(branch_consumers) < 2:
                continue
            for reshape in [node for node in branch_consumers if node.op_type == "Reshape" and node.output]:
                reshape_qs = [node for node in consumers.get(reshape.output[0], []) if node.op_type == "QuantizeLinear"]
                if len(reshape_qs) != 1:
                    continue
                reshape_q = reshape_qs[0]
                reshape_dqs = [
                    node for node in consumers.get(reshape_q.output[0], []) if node.op_type == "DequantizeLinear"
                ]
                if not reshape_dqs:
                    continue
                reshape_params = quant_params(reshape_q)
                if reshape_params is None:
                    continue
                reshape_scale, reshape_zp = reshape_params
                if split_scale.dtype != reshape_scale.dtype or split_zp.dtype != reshape_zp.dtype:
                    continue
                if np.array_equal(split_scale, reshape_scale) and np.array_equal(split_zp, reshape_zp):
                    continue

                reshape_lo, reshape_hi = quant_range(reshape_scale, reshape_zp)
                tolerance = max(abs(split_lo), abs(split_hi), 1.0) * 1e-7
                if split_lo > reshape_lo + tolerance or split_hi < reshape_hi - tolerance:
                    continue

                base = reshape.name.replace("/", "_").replace(".", "_") or "reshape"
                scale_name = unique_name(f"{base}_split_input_scale")
                zp_name = unique_name(f"{base}_split_input_zero_point")
                scale_init = onnx.numpy_helper.from_array(split_scale.copy(), scale_name)
                zp_init = onnx.numpy_helper.from_array(split_zp.copy(), zp_name)
                graph.initializer.extend([scale_init, zp_init])
                initializers[scale_name] = scale_init
                initializers[zp_name] = zp_init

                for node in [reshape_q, *reshape_dqs]:
                    node.input[1] = scale_name
                    node.input[2] = zp_name

                print(
                    f"  split-reshape quant @ {reshape.name} <- {split.name}: "
                    f"range [{reshape_lo:.4f}, {reshape_hi:.4f}] -> [{split_lo:.4f}, {split_hi:.4f}]"
                )
                aligned += 1

    return aligned


def export_onnx(
    export_wrapper: torch.nn.Module,
    inputs: torch.Tensor,
    out_path: str,
    output_names: list[str],
    fix_split_reshape_quant: bool = True,
) -> None:
    onnx_program = torch.onnx.export(
        export_wrapper,
        (inputs,),
        dynamo=True,
        opset_version=21,
        output_names=output_names,
    )
    onnx_program.optimize()
    onnx_program.save(out_path)
    print(f"export qat model to [{out_path}] done!")

    model_simp = slim(onnx.load(out_path))
    ensure_main_opset(model_simp, REQUIRED_ONNX_OPSET)
    removed_outputs = remove_invalid_graph_outputs(model_simp)
    n_merged_ne = _merge_near_equal_requant(model_simp)
    if n_merged_ne:
        print(
            f"merged {n_merged_ne} near-equal requant (严格: scale<2% + |zp|<=1 + dtype一致 + q单消费者, fan-out安全)"
        )
    n_fixed = _fix_qdq_qdq_mismatch(model_simp)
    if n_fixed:
        print(f"inserted {n_fixed} requant marker(s) where Q→DQ→Q scales diverge")
    n_split = _split_shared_mul_dequant(model_simp)
    if n_split:
        print(f"split {n_split} shared DequantizeLinear(s) for independent Mul ops")
    n_merged = _merge_adjacent_dq_q(model_simp)
    if n_merged:
        print(f"merged {n_merged} redundant DQ→Q pair(s) with near-identical scales")
    if fix_split_reshape_quant:
        n_aligned = _align_split_reshape_quantization(model_simp)
        if n_aligned:
            print(f"aligned {n_aligned} Split fan-out Reshape quantization range(s) for AXERA NPU")
    sim_path = out_path.replace(".onnx", "_slim.onnx")
    onnx.save(model_simp, sim_path)
    onnx.checker.check_model(model_simp)
    if removed_outputs:
        print(f"remove invalid slim outputs: {removed_outputs}")
    print(f"save onnx model to [{sim_path}] Successfully!")


def main() -> None:
    args = parse_args()
    cfg = resolve_paths(args)
    qat_onnx_imgsz = normalize_imgsz(args.imgsz)
    output_path = str(Path(cfg.out))

    print(
        f"task={cfg.task}, model={cfg.model}, pretrained={cfg.pretrained}, "
        f"qat_weights={cfg.qat_weights}, out={output_path}"
    )

    quantized_model, inputs = build_quantized_model(
        cfg,
        args.quant_config,
        args.device,
        qat_onnx_imgsz,
        export_pth=args.export_pth,
    )
    export_wrapper, output_names = build_export_plan(cfg.task, quantized_model, inputs, end2end=args.end2end)
    export_onnx(
        export_wrapper,
        inputs,
        output_path,
        output_names,
        fix_split_reshape_quant=args.fix_split_reshape_quant,
    )


if __name__ == "__main__":
    main()
# python export.py --task segment
# codex resume 019d0515-c4ad-7991-9fdc-a578b0446277
