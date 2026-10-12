#!/usr/bin/env python3
"""YOLO PT2E QAT/PTQ 损失归因：backbone / neck / head 分段消融与逐层误差。

复用 eval.py qat 的 PT2E 评估口径：
  prepare_pt2e_qat_model 重建同构图 -> 严格加载 QAT checkpoint 的 qat_model state ->
  冻结 observer（保留 fake quant）-> DetectionValidator 全量 COCO val。
若 S0（check_export_graph_equivalence.py）报分叉，先修模型代码并重训，本脚本结论才有效。

阶段：
  map          fake-quant -> 段（backbone/neck/head）映射与覆盖率
  groups       QAT checkpoint 分段 fake-quant 消融（残余舍入）
  ptq_groups   浮点权重 + 校准分段消融（容量损失）
  numeric      浮点图 vs QAT 的 feats/boxes/scores SNR
  layerwise    逐 conv SNR 与最差层
  weights      分段权重 per-channel range/std 统计

用法：
  python .codex/skills/qat-accuracy-attribution/scripts/qat_loss_attribution.py --stage groups --device 0 --batch 32
  python .codex/skills/qat-accuracy-attribution/scripts/qat_loss_attribution.py --stage numeric --device 0 --sample 100
  python .codex/skills/qat-accuracy-attribution/scripts/qat_loss_attribution.py --stage weights
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import argparse
import copy
import json
import warnings
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.export import Dim

from ultralytics import YOLO
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.qat_utils import prepare_pt2e_qat_model
from ultralytics.utils.torch_utils import select_device
import ultralytics.utils.quantized_decomposed_dequantize_per_channel  # noqa: F401

warnings.filterwarnings("ignore")

SEGMENTS = ("backbone", "neck", "head")


class FakeTrainer:
    """把 QAT prepared graph 喂给 DetectionValidator 的 training-val 路径。"""

    def __init__(self, float_model, qat_model, data_dict, device, end2end, task="detect"):
        self.model = float_model
        self.qat_model = qat_model
        self.device = device
        self.data = data_dict
        self.ema = None
        self.amp = False
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        self.loss_items = torch.zeros(len(self.loss_names))
        self.world_size = 1
        self.epoch = 0
        self.epochs = 1
        self.stopper = SimpleNamespace(possible_stop=False)
        self.args = argparse.Namespace(
            half=False, amp=False, compile=False, plots=False, end2end=end2end,
            conf=0.001, iou=0.7, max_det=300, single_cls=False, agnostic_nms=False,
            save_json=False, save_hybrid=False,
        )

    def label_loss_items(self, loss_items=None, prefix="val"):
        if loss_items is not None:
            return dict(zip([f"{prefix}/{x}" for x in self.loss_names], [round(float(x), 5) for x in loss_items]))
        return [f"{prefix}/{x}" for x in self.loss_names]


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv10s U8-QAT backbone/neck/head loss attribution")
    p.add_argument("--stage", choices=("map", "groups", "ptq_groups", "numeric", "layerwise", "weights", "all"), default="groups")
    p.add_argument("--calib", type=int, default=500, help="Calibration images for ptq_groups.")
    p.add_argument("--ckpt", default="runs/detect/yolov10s-qat-siluInU8-attnS8/weights/best.pt")
    p.add_argument("--model", default="yolov10s.yaml")
    p.add_argument("--pretrained", default="weights/yolov10s.pt")
    p.add_argument("--quant-config", dest="quant_config", default="config-qat/config_yolov10s_siluInU8_attnS8.json")
    p.add_argument("--data", default="coco.yaml")
    p.add_argument("--device", default="2")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Only run first N val images per group (0=all).")
    p.add_argument("--sample", type=int, default=100, help="Images for numeric error stage.")
    p.add_argument("--out", default="cache/debug/qat_loss_attribution.json")
    return p.parse_args()


def build_models(a, device):
    data_dict = check_det_dataset(a.data)
    model = YOLO(a.model, task="detect")
    model.load(a.pretrained)
    float_model = model.model.float().to(device)
    float_model.nc = data_dict["nc"]
    float_model.names = data_dict["names"]
    float_model.train()
    float_model.model[-1].end2end = True

    example = torch.rand(2, 3, a.imgsz, a.imgsz, device=device)
    program = torch.export.export_for_training(
        float_model, (example,), dynamic_shapes={"x": {0: Dim.AUTO, 2: Dim.AUTO, 3: Dim.AUTO}},
    )
    float_graph = program.module().to(device)
    torch.ao.quantization.move_exported_model_to_eval(float_graph)

    _, prepared = prepare_pt2e_qat_model(
        float_model=float_model, device=device, config_path=a.quant_config,
        imgsz=a.imgsz, dynamic_batch_max=128,
    )
    model_args = getattr(float_model, "args", {})
    hyp = dict(DEFAULT_CFG_DICT, **model_args) if isinstance(model_args, dict) else dict(DEFAULT_CFG_DICT)
    hyp.setdefault("box", 7.5)
    hyp.setdefault("cls", 0.5)
    hyp.setdefault("dfl", 1.5)
    float_model.args = SimpleNamespace(**hyp)
    float_model.criterion = float_model.init_criterion()

    ckpt = torch.load(a.ckpt, weights_only=False, map_location="cpu")
    qat_state = ckpt.get("qat_ema") or ckpt.get("qat_model") or ckpt.get("model")
    assert qat_state is not None, "checkpoint has no qat_ema/qat_model/model state"
    rep = prepared.load_state_dict(qat_state, strict=False)
    assert not rep.missing_keys, f"observer mismatch: missing={len(rep.missing_keys)}"
    print(f"[load] missing={len(rep.missing_keys)} unexpected={len(rep.unexpected_keys)}", flush=True)

    prepared = BaseValidator._prepare_pt2e_model_for_eval(prepared)
    prepared.apply(torch.ao.quantization.disable_observer)
    prepared.to(device)
    prepared.eval()
    return float_model, float_graph, prepared, data_dict


def is_fake_quant_node(node):
    return node.op == "call_module" and "activation_post_process" in str(node.target)


def _segment_of_conv(node):
    stack = node.meta.get("nn_module_stack") or {}
    for value in stack.values():
        path = value[0] if isinstance(value, (tuple, list)) else value
        if isinstance(path, str) and path.startswith("model."):
            index = path.split(".")[1]
            if index.isdigit():
                i = int(index)
                return "backbone" if i <= 10 else ("neck" if i <= 22 else "head")
    return None


def _upstream_segments(node):
    queue = deque([(arg, 0) for arg in node.args if isinstance(arg, torch.fx.Node)])
    seen, segs = set(), set()
    while queue:
        current, depth = queue.popleft()
        if id(current) in seen or depth > 120:
            continue
        seen.add(id(current))
        if current.op == "call_function" and current.target == torch.ops.aten.conv2d.default:
            seg = _segment_of_conv(current)
            if seg:
                segs.add(seg)
            continue
        queue.extend((arg, depth + 1) for arg in current.args if isinstance(arg, torch.fx.Node))
    return segs


def _downstream_segments(node):
    segs = set()
    for user in node.users:
        if user.op == "call_function" and user.target == torch.ops.aten.conv2d.default:
            seg = _segment_of_conv(user)
            if seg:
                segs.add(seg)
    return segs


def map_fq_to_segments(prepared):
    """Return {segment: [module names]} for every fake-quant module.

    Activation fake-quant are assigned by their upstream conv; weight fake-quant
    (fed by ``get_attr``) are assigned by the conv consuming them.
    """
    fq_nodes = [n for n in prepared.graph.nodes if is_fake_quant_node(n)]
    mapping = {seg: [] for seg in SEGMENTS}
    unresolved = []
    for node in fq_nodes:
        segs = _upstream_segments(node) or _downstream_segments(node)
        if not segs:
            mapping["backbone"].append(node.target)
            unresolved.append(node.name)
        elif segs == {"backbone"}:
            mapping["backbone"].append(node.target)
        elif "head" in segs:
            mapping["head"].append(node.target)
        else:
            mapping["neck"].append(node.target)
    print(f"[map] fake-quant total={len(fq_nodes)}")
    for seg in SEGMENTS:
        print(f"      {seg}: {len(mapping[seg])}")
    print(f"      no-upstream/downstream conv (assigned backbone): {len(unresolved)}")
    return mapping, fq_nodes


def set_segment_fake_quant(prepared, mapping, enabled_segments):
    for seg in SEGMENTS:
        enabled = seg in enabled_segments
        for name in mapping[seg]:
            module = prepared.get_submodule(name)
            if hasattr(module, "fake_quant_enabled"):
                module.fake_quant_enabled.fill_(1 if enabled else 0)


def build_val_loader(a, data_dict, batch=None, limit=0, split="val"):
    gs = 32
    val_ns = argparse.Namespace(
        task="detect", data=a.data, imgsz=a.imgsz, batch=batch or a.batch, workers=a.workers, fraction=1.0,
        augment=False, erasing=0.0, flipud=0.0, fliplr=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0,
        translate=0.0, scale=0.0, shear=0.0, perspective=0.0, mosaic=0.0, mixup=0.0, cutmix=0.0,
        copy_paste=0.0, auto_augment=None, single_cls=False, classes=None, overlap_mask=False, mask_ratio=4,
        rect=False, cache=False,
    )
    dataset = build_yolo_dataset(val_ns, data_dict[split], batch or a.batch, data_dict, mode="val", rect=False, stride=gs)
    if limit > 0:
        dataset.labels = dataset.labels[:limit]
        dataset.im_files = dataset.im_files[:limit]
        dataset.ims = [None] * len(dataset.labels)
        dataset.npy_files = dataset.npy_files[:limit]
    return build_dataloader(dataset, batch=batch or a.batch, workers=a.workers, shuffle=False, rank=-1, drop_last=False)


def run_val(prepared, float_model, data_dict, a, device, tag, limit=0):
    loader = build_val_loader(a, data_dict, limit=limit)
    cfg = copy.deepcopy(DEFAULT_CFG_DICT)
    cfg.update({
        "task": "detect", "mode": "val", "data": a.data, "imgsz": a.imgsz, "batch": a.batch,
        "device": a.device, "workers": a.workers, "split": "val", "end2end": True, "conf": 0.001,
        "iou": 0.7, "max_det": 300, "half": False, "plots": False, "save_json": False, "save_hybrid": False,
    })
    validator = DetectionValidator(dataloader=loader, args=cfg)
    trainer = FakeTrainer(float_model, prepared, data_dict, device, True)
    results = validator(trainer=trainer)
    metrics = {
        "mAP50-95": float(results.get("metrics/mAP50-95(B)", 0.0)),
        "mAP50": float(results.get("metrics/mAP50(B)", 0.0)),
    }
    print(f"[val:{tag}] mAP50-95={metrics['mAP50-95']:.4f} mAP50={metrics['mAP50']:.4f}", flush=True)
    return metrics


def stage_map(a, device):
    _, _, prepared, data_dict = build_models(a, device)
    mapping, fq_nodes = map_fq_to_segments(prepared)
    details = {}
    for seg in SEGMENTS:
        samples = []
        for name in mapping[seg][:8]:
            node = next(n for n in fq_nodes if n.target == name)
            samples.append({
                "fq": name,
                "upstream": sorted(_upstream_segments(node)),
                "downstream": sorted(_downstream_segments(node)),
                "input_ops": sorted({str(getattr(arg, "op", "scalar")) for arg in node.args}),
            })
        details[seg] = samples
    names_out = {seg: list(names) for seg, names in mapping.items()}
    return {"fq_total": len(fq_nodes), "by_segment": {k: len(v) for k, v in mapping.items()}}, names_out, details


def stage_groups(a, device):
    float_model, _, prepared, data_dict = build_models(a, device)
    mapping, _ = map_fq_to_segments(prepared)
    groups = {
        "baseline": set(SEGMENTS),
        "-backbone": {"neck", "head"},
        "-neck": {"backbone", "head"},
        "-head": {"backbone", "neck"},
        "-all": set(),
    }
    results = {}
    for tag, enabled in groups.items():
        set_segment_fake_quant(prepared, mapping, enabled)
        results[tag] = run_val(prepared, float_model, data_dict, a, device, tag, limit=a.limit)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"groups": results}, f, ensure_ascii=False, indent=2)
    set_segment_fake_quant(prepared, mapping, set(SEGMENTS))
    return results


def _compare(ref, out, tag, rows):
    ref = ref.float()
    out = out.float()
    diff = out - ref
    ref_norm = ref.flatten().norm()
    snr = 10 * torch.log10((ref_norm**2) / (diff.flatten().norm() ** 2 + 1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), out.flatten(), dim=0).item()
    rel = (diff.abs().max() / (ref.abs().max() + 1e-12)).item()
    rows.append({"tensor": tag, "shape": list(ref.shape), "snr_db": round(snr, 2),
                 "cosine": round(cos, 6), "rel_max": round(rel, 5)})


def stage_numeric(a, device):
    _, exported, prepared, data_dict = build_models(a, device)
    loader = build_val_loader(a, data_dict, batch=8)
    rows = []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index * 8 >= a.sample:
                break
            images = batch["img"].to(device).float() / 255.0
            f_out = exported(images)
            f_preds = f_out[1] if isinstance(f_out, tuple) else f_out
            f_o2o = f_preds["one2one"]
            q_out = prepared(images)
            q_o2o = q_out["one2one"]
            for level, (fr, qr) in enumerate(zip(f_o2o["feats"], q_o2o["feats"])):
                _compare(fr, qr, f"feat_p{3 + level}", rows)
            for level, (fb, qb) in enumerate(zip(f_o2o["boxes"], q_o2o["boxes"])):
                _compare(fb, qb, f"boxes_p{3 + level}", rows)
            for level, (fs, qs) in enumerate(zip(f_o2o["scores"], q_o2o["scores"])):
                _compare(fs, qs, f"scores_p{3 + level}", rows)
    agg = {}
    for row in rows:
        key = row["tensor"]
        agg.setdefault(key, []).append(row)
    summary = {
        key: {
            "snr_db": round(sum(r["snr_db"] for r in vals) / len(vals), 2),
            "cosine": round(sum(r["cosine"] for r in vals) / len(vals), 6),
            "rel_max": round(max(r["rel_max"] for r in vals), 5),
        }
        for key, vals in agg.items()
    }
    return {"summary": summary, "per_batch": rows}


def stage_weights(a):
    result = {}
    for name, weights in (("yolov10s", a.pretrained), ("yolo26s", "weights/yolo26s.pt")):
        if not Path(weights).is_file():
            continue
        model = YOLO(weights).model.float()
        stats = {seg: {"convs": 0, "range_mean": 0.0, "std_mean": 0.0} for seg in SEGMENTS}
        acc = {seg: {"range": [], "std": []} for seg in SEGMENTS}
        head = model.model[-1]
        layers = list(model.model)
        for i, layer in enumerate(layers):
            seg = "backbone" if i <= 10 else ("neck" if i <= 22 else "head")
            for module in layer.modules():
                if isinstance(module, torch.nn.Conv2d):
                    w = module.weight.detach()
                    per_channel = w.view(w.shape[0], -1)
                    rng = (per_channel.max(dim=1).values - per_channel.min(dim=1).values).mean().item()
                    acc[seg]["range"].append(rng)
                    acc[seg]["std"].append(w.std().item())
        for seg in SEGMENTS:
            n = len(acc[seg]["range"])
            stats[seg] = {
                "convs": n,
                "range_mean": round(sum(acc[seg]["range"]) / n, 4) if n else 0.0,
                "std_mean": round(sum(acc[seg]["std"]) / n, 4) if n else 0.0,
            }
        result[name] = stats
    return result


def stage_layerwise(a, device):
    _, float_graph, prepared, data_dict = build_models(a, device)
    loader = build_val_loader(a, data_dict, batch=4)
    from torch.fx import Interpreter

    float_convs = [
        n for n in float_graph.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.conv2d.default
    ]
    prepared_convs = [
        n for n in prepared.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.conv2d.default
    ]
    count = min(len(float_convs), len(prepared_convs))
    print(f"[layerwise] float convs={len(float_convs)} prepared convs={len(prepared_convs)}", flush=True)
    print(f"[layerwise] float names[:6]={[n.name for n in float_convs[:6]]}", flush=True)
    print(f"[layerwise] prep  names[:6]={[n.name for n in prepared_convs[:6]]}", flush=True)
    seg_of = {i: _segment_of_conv(float_convs[i]) for i in range(count)}

    class Collect(Interpreter):
        def __init__(self, graph_module, store):
            super().__init__(graph_module)
            self.store = store
            self._order = []

        def run_node(self, node):
            result = super().run_node(node)
            if node.op == "call_function" and node.target == torch.ops.aten.conv2d.default:
                if isinstance(result, torch.Tensor) and len(self._order) < count:
                    self.store[len(self._order)] = result.detach()
                self._order.append(node.name)
            return result

    samples = {}
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index * 4 >= a.sample:
                break
            images = batch["img"].to(device).float() / 255.0
            float_store, qat_store = {}, {}
            Collect(float_graph, float_store).run(images)
            Collect(prepared, qat_store).run(images)
            for i in range(count):
                ref, out = float_store.get(i), qat_store.get(i)
                if ref is None or out is None:
                    continue
                ref = ref.float()
                diff = out.float() - ref
                snr = 10 * torch.log10(ref.flatten().norm() ** 2 / (diff.flatten().norm() ** 2 + 1e-12)).item()
                samples.setdefault(i, []).append(snr)

    name_to_index = {float_convs[i].name: i for i in range(count)}
    per_node = {float_convs[i].name: round(sum(vals) / len(vals), 2) for i, vals in samples.items()}
    per_seg = {}
    for seg in SEGMENTS:
        vals = sorted(v for name, v in per_node.items() if seg_of.get(name_to_index[name]) == seg)
        if vals:
            per_seg[seg] = {
                "convs": len(vals),
                "snr_median": round(vals[len(vals) // 2], 2),
                "snr_min": vals[0],
                "snr_max": vals[-1],
            }
    worst = sorted(per_node.items(), key=lambda kv: kv[1])[:15]
    return {
        "per_segment": per_seg,
        "worst_convs": [{"node": k, "segment": seg_of.get(name_to_index[k]), "snr": v} for k, v in worst],
        "per_node": per_node,
    }


def build_ptq_models(a, device, calib):
    """Float weights + PTQ calibration (no QAT checkpoint) for capacity-loss attribution."""
    data_dict = check_det_dataset(a.data)
    model = YOLO(a.model, task="detect")
    model.load(a.pretrained)
    float_model = model.model.float().to(device)
    float_model.nc = data_dict["nc"]
    float_model.names = data_dict["names"]
    float_model.train()
    float_model.model[-1].end2end = True

    example = torch.rand(2, 3, a.imgsz, a.imgsz, device=device)
    program = torch.export.export_for_training(
        float_model, (example,), dynamic_shapes={"x": {0: Dim.AUTO, 2: Dim.AUTO, 3: Dim.AUTO}},
    )
    float_graph = program.module().to(device)
    torch.ao.quantization.move_exported_model_to_eval(float_graph)

    _, prepared = prepare_pt2e_qat_model(
        float_model=float_model, device=device, config_path=a.quant_config,
        imgsz=a.imgsz, dynamic_batch_max=128,
    )
    prepared = BaseValidator._prepare_pt2e_model_for_eval(prepared)
    prepared.to(device)
    prepared.eval()
    calib_loader = build_val_loader(a, data_dict, batch=16, limit=calib, split="train")
    with torch.no_grad():
        for batch in calib_loader:
            prepared(batch["img"].to(device).float() / 255.0)
    prepared.apply(torch.ao.quantization.disable_observer)

    model_args = getattr(float_model, "args", {})
    hyp = dict(DEFAULT_CFG_DICT, **model_args) if isinstance(model_args, dict) else dict(DEFAULT_CFG_DICT)
    hyp.setdefault("box", 7.5)
    hyp.setdefault("cls", 0.5)
    hyp.setdefault("dfl", 1.5)
    float_model.args = SimpleNamespace(**hyp)
    float_model.criterion = float_model.init_criterion()
    return float_model, float_graph, prepared, data_dict


def stage_ptq_groups(a, device):
    float_model, _, prepared, data_dict = build_ptq_models(a, device, calib=a.calib)
    mapping, _ = map_fq_to_segments(prepared)
    groups = {
        "ptq-baseline": set(SEGMENTS),
        "ptq-backbone-fp32": {"neck", "head"},
        "ptq-neck-fp32": {"backbone", "head"},
        "ptq-head-fp32": {"backbone", "neck"},
        "ptq-all-fp32": set(),
    }
    results = {}
    for tag, enabled in groups.items():
        set_segment_fake_quant(prepared, mapping, enabled)
        results[tag] = run_val(prepared, float_model, data_dict, a, device, tag, limit=a.limit)
        out_path = Path(a.out).parent / (Path(a.out).stem + "_ptq.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"groups": results}, f, ensure_ascii=False, indent=2)
    set_segment_fake_quant(prepared, mapping, set(SEGMENTS))
    return results


def main():
    a = parse_args()
    device = select_device(a.device)
    payload = {}
    if a.stage in ("map", "all"):
        summary, names, details = stage_map(a, device)
        payload["map_summary"] = summary
        payload["map_names"] = names
        payload["map_details"] = details
    if a.stage in ("groups", "all"):
        payload["groups"] = stage_groups(a, device)
    if a.stage == "ptq_groups":
        payload["ptq_groups"] = stage_ptq_groups(a, device)
    if a.stage in ("numeric", "all"):
        payload["numeric"] = stage_numeric(a, device)
    if a.stage in ("layerwise", "all"):
        payload["layerwise"] = stage_layerwise(a, device)
    if a.stage in ("weights", "all"):
        payload["weights"] = stage_weights(a)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[done] saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
