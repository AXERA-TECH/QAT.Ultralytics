#!/usr/bin/env python3
"""S0：核查 eager 模型与 torch.export.export_for_training 训练图的一致性。

dynamo 导出可能错误追踪训练图并静默改变网络语义，例如 C2f.forward 的
``y.extend(m(y[-1]) for m in self.m)`` 列表突变被缓存为初值（bottleneck 并行化）。
这类偏差会污染所有后续 QAT/PTQ 分析结论，必须在归因实验之前先做本检查。
不一致时应在模型代码中修复（参考 block.py C2f 的显式循环写法）并从浮点权重重训。

用法：
  python .codex/skills/qat-accuracy-attribution/scripts/check_export_graph_equivalence.py \
    --model yolov10s.yaml --pretrained weights/yolov10s.pt --device cuda:0
  python .codex/skills/qat-accuracy-attribution/scripts/check_export_graph_equivalence.py \
    --model yolov10s.yaml --pretrained weights/yolov10s.pt --device cuda:0 \
    --c2f-internal --c2f-layers 2,4

退出码：0 = 图与 eager 一致；1 = 检测到分叉。
"""

import argparse
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
import torch.ao.quantization as aoq
from torch.export import Dim
from torch.fx import Interpreter

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Compare eager model against the export_for_training graph.")
    parser.add_argument("--model", required=True, help="Model yaml, e.g. yolov10s.yaml")
    parser.add_argument("--pretrained", default=None, help="Optional float pretrained weights.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Max-abs diff treated as equal.")
    parser.add_argument("--layers", default="", help="Comma-separated layer indices to compare (default: all).")
    parser.add_argument("--c2f-internal", action="store_true", help="Drill into C2f cv1/m*/cv2.")
    parser.add_argument(
        "--c2f-layers", default="", help="Comma-separated C2f layer indices for internal check (default: first diverging)."
    )
    return parser.parse_args()


def stack_paths(node):
    stack = node.meta.get("nn_module_stack") or {}
    return [value[0] if isinstance(value, (tuple, list)) else value for value in stack.values()]


def _path_matches(paths, prefix):
    for path in paths:
        if not isinstance(path, str):
            continue
        if path == prefix or path.startswith(prefix + "."):
            return True
    return False


def last_node_by_prefix(graph, prefix):
    found = None
    for node in graph.graph.nodes:
        if node.op != "call_function":
            continue
        if _path_matches(stack_paths(node), prefix):
            found = node
    return found


class GraphCollector(Interpreter):
    """Run the graph and record node outputs by key."""

    def __init__(self, graph_module, key_by_node):
        super().__init__(graph_module)
        self.key_by_node = key_by_node
        self.store = {}

    def run_node(self, node):
        result = super().run_node(node)
        if node.name in self.key_by_node and isinstance(result, torch.Tensor):
            self.store[self.key_by_node[node.name]] = result.detach()
        return result


def diff_of(reference, output):
    if reference.shape != output.shape:
        return None, f"shape {tuple(reference.shape)} vs {tuple(output.shape)}"
    return (reference.float() - output.float()).abs().max().item(), ""


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cpu" else "cpu")

    model = YOLO(args.model)
    if args.pretrained:
        model.load(args.pretrained)
    detection = model.model.float().to(device)
    detection.train()
    layers = list(detection.model)
    layer_indices = list(range(len(layers))) if not args.layers else [int(v) for v in args.layers.split(",")]

    x = torch.rand(1, 3, args.imgsz, args.imgsz, device=device)
    program = torch.export.export_for_training(
        detection, (x,), dynamic_shapes={"x": {0: Dim.AUTO, 2: Dim.AUTO, 3: Dim.AUTO}},
    )
    graph = program.module().to(device)
    aoq.move_exported_model_to_eval(graph)

    eager_outputs = {}

    def hook(module, inputs, output, idx=None):
        if isinstance(output, torch.Tensor):
            eager_outputs[idx] = output.detach()

    handles = [layer.register_forward_hook(lambda mod, i, o, idx=i: hook(mod, i, o, idx)) for i, layer in enumerate(layers)]
    detection.eval()
    with torch.no_grad():
        detection(x)
    for handle in handles:
        handle.remove()

    key_by_node = {}
    for idx in layer_indices:
        node = last_node_by_prefix(graph, f"model.{idx}")
        if node is not None:
            key_by_node[node.name] = idx
    with torch.no_grad():
        collector = GraphCollector(graph, key_by_node)
        collector.run(x)

    failures = []
    first_divergence = None
    print("layer | max_abs_diff")
    for idx in layer_indices:
        if idx not in eager_outputs or idx not in collector.store:
            print(f"{idx:5d} | skip (non-tensor output or no graph node)")
            continue
        diff, note = diff_of(eager_outputs[idx], collector.store[idx])
        if diff is None:
            print(f"{idx:5d} | {note}")
            failures.append(idx)
            first_divergence = first_divergence if first_divergence is not None else idx
            continue
        flag = "" if diff <= args.tolerance else "  <-- DIVERGED"
        print(f"{idx:5d} | {diff:.6f}{flag}")
        if diff > args.tolerance:
            failures.append(idx)
            if first_divergence is None:
                first_divergence = idx

    if first_divergence is not None:
        layer = layers[first_divergence]
        print(f"[first divergence] layer {first_divergence} ({type(layer).__name__})")

    if args.c2f_internal:
        if args.c2f_layers:
            internal_layers = [int(v) for v in args.c2f_layers.split(",")]
        else:
            internal_layers = [first_divergence] if first_divergence is not None else []
        for idx in internal_layers:
            if idx is None or idx >= len(layers):
                continue
            layer = layers[idx]
            if not hasattr(layer, "m") or not hasattr(layer, "cv1"):
                print(f"[c2f-internal] layer {idx} ({type(layer).__name__}) has no C2f structure, skip")
                continue
            internal_eager = {}
            handles = [
                layer.cv1.register_forward_hook(
                    lambda mod, i, o, key=(idx, "cv1"): internal_eager.__setitem__(key, o.detach())
                ),
                layer.cv2.register_forward_hook(
                    lambda mod, i, o, key=(idx, "cv2"): internal_eager.__setitem__(key, o.detach())
                ),
            ]
            for j, block in enumerate(layer.m):
                handles.append(
                    block.register_forward_hook(
                        lambda mod, i, o, key=(idx, f"m{j}"): internal_eager.__setitem__(key, o.detach())
                    )
                )
            with torch.no_grad():
                detection(x)
            for handle in handles:
                handle.remove()

            internal_keys = {}
            for name in ("cv1", "cv2"):
                node = last_node_by_prefix(graph, f"model.{idx}.{name}")
                if node is not None:
                    internal_keys[node.name] = (idx, name)
            for j in range(len(layer.m)):
                node = last_node_by_prefix(graph, f"model.{idx}.m.{j}")
                if node is not None:
                    internal_keys[node.name] = (idx, f"m{j}")
            with torch.no_grad():
                internal = GraphCollector(graph, internal_keys)
                internal.run(x)
            print(f"[c2f-internal] layer {idx} ({type(layer).__name__})")
            for key in sorted(internal_eager):
                output = internal.store.get(key)
                if output is None:
                    print(f"  {key[1]:>4s} | no graph match")
                    continue
                diff, note = diff_of(internal_eager[key], output)
                text = note if diff is None else f"{diff:.6f}"
                flag = "" if diff is not None and diff <= args.tolerance else "  <-- DIVERGED"
                print(f"  {key[1]:>4s} | {text}{flag}")
                if diff is None or diff > args.tolerance:
                    failures.append(f"layer{idx}.{key[1]}")

    if failures:
        print(f"\nRESULT: DIVERGED ({len(failures)} item(s): {failures[:8]})")
        return 1
    print("\nRESULT: PASS (eager and export_for_training graphs match)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
