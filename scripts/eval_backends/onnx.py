#!/usr/bin/env python3
"""直接评测导出的 QAT ONNX（6 输出 raw one2one）的 mAP。
onnxruntime 跑 onnx → 重组 {"one2one":{"boxes":[p3,p4,p5],"scores":[...]}} →
与 eval_convert 完全同一条 validator 链（_rebuild_pt2e_predictions → head._inference + postprocess）。
用途：补齐 convert(pytorch) → onnx(ort) → NPU 链条的中间实测环。
注意：交付 onnx batch 固定 1 → 逐图 ort 推理（CPU EP，5000 图约 15-30 分钟）。.

用法（2026-07-13 实测指令，exp50 交付 onnx）：
  ssh qat-dev 'cd /home/heqi/project-qat/ultralytics && \
    PYTHONPATH=/home/heqi/project-qat/ultralytics \
    /home/heqi/miniforge3/envs/torch2.6-qat-yolo/bin/python -u eval.py onnx \
      --onnx yolo26_onnx/exp50_merged2_slim.onnx \
      --end2end True --rect False --pycoco True --device cuda:3 \
      > /tmp/eval_onnx50.log 2>&1'

实测结果（exp50_merged2_slim.onnx，rect=False square，全 5000 val，~15min）：
  ultralytics-metric  mAP50-95=0.3917  mAP50=0.5501
  pycocotools/COCO    mAP50-95=0.3933（AP50=0.555）
  ——与 prepared 冻结 0.3938 / convert 0.3924 差 ±0.001 后端噪声级（框架链健康）；
    NPU 板端 0.3907 → onnx→NPU 缺口 -0.26 点（pulsar2 编译/板端/管线），见
    todos/work/20260713-npu-deploy-gap/analysis.md §2。
"""

import argparse
import copy
import json
import warnings
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch

from ultralytics import YOLO
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.torch_utils import select_device

warnings.filterwarnings("ignore")


class FakeTrainer:
    def __init__(self, float_model, qat_model, data_dict, device, end2end):
        self.model = float_model
        self.qat_model = qat_model
        self.device = device
        self.data = data_dict
        self.ema = None
        self.amp = False
        self.loss_items = torch.zeros(3)
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss")
        self.world_size = 1
        self.epoch = 0
        self.epochs = 1
        from collections import namedtuple

        self.stopper = namedtuple("S", ["possible_stop"])(possible_stop=False)
        self.args = argparse.Namespace(
            half=False,
            amp=False,
            compile=False,
            plots=False,
            end2end=end2end,
            conf=0.001,
            iou=0.7,
            max_det=300,
            single_cls=False,
            agnostic_nms=False,
            save_json=False,
            save_hybrid=False,
        )

    def label_loss_items(self, loss_items=None, prefix="val"):
        if loss_items is not None:
            return dict(zip([f"{prefix}/{x}" for x in self.loss_names], [round(float(x), 5) for x in loss_items]))
        return [f"{prefix}/{x}" for x in self.loss_names]


class OrtOne2One(torch.nn.Module):
    """把六输出 raw ONNX 包装为 validator 所需的 PT2E dict（batch 固定为 1，逐图执行）。."""

    def __init__(self, onnx_path, device, box_channels, score_channels, end2end):
        super().__init__()
        so = ort.SessionOptions()
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.dev = device
        self.box_channels = box_channels
        self.score_channels = score_channels
        self.end2end = end2end

    def forward(self, x):
        per_img = [self.sess.run(None, {self.iname: x[i : i + 1].detach().cpu().numpy()}) for i in range(x.shape[0])]
        boxes, scores = {}, {}
        for pos_outs in zip(*per_img):  # 同一输出位置跨 batch 拼
            t = torch.from_numpy(np.concatenate(pos_outs, 0)).to(self.dev)
            if t.shape[1] == self.box_channels:
                boxes[t.shape[2]] = t
            elif t.shape[1] == self.score_channels:
                scores[t.shape[2]] = t
            else:
                raise RuntimeError(
                    f"Unexpected ONNX output shape {tuple(t.shape)}; expected "
                    f"box channels={self.box_channels} or score channels={self.score_channels}"
                )
        if boxes.keys() != scores.keys() or len(boxes) != 3:
            raise RuntimeError("Expected three matching box/score outputs from the raw one2one ONNX model")
        bl = [boxes[k] for k in sorted(boxes, reverse=True)]  # anchor 6400/1600/400 = p3/p4/p5
        sl = [scores[k] for k in sorted(scores, reverse=True)]
        # head._get_decode_boxes 只用 feats 的 shape/dtype/device 生成 anchors → dummy 即可（80/40/20 由 anchor 数反推）
        B = x.shape[0]
        feats = [torch.zeros(B, 1, int(k**0.5), int(k**0.5), device=self.dev) for k in sorted(boxes, reverse=True)]
        pred_dict = {"boxes": bl, "scores": sl, "feats": feats}
        if not self.end2end:
            return pred_dict
        # one2many 用同批张量的副本占位，仅为 E2EDetectLoss 不崩（loss 数值无意义，不影响 mAP）
        one2many = {
            "boxes": [b.clone() for b in bl],
            "scores": [s.clone() for s in sl],
            "feats": [f.clone() for f in feats],
        }
        return {"one2one": pred_dict, "one2many": one2many}


p = argparse.ArgumentParser()
p.add_argument("--onnx", required=True, help="导出的 QAT onnx（6 输出 raw one2one）")
p.add_argument("--model", default="yolo26n.yaml")
p.add_argument("--pretrained", default="yolo26n.pt")
p.add_argument("--data", default="coco.yaml")
p.add_argument("--end2end", default="True")
p.add_argument("--rect", default="False")
p.add_argument("--pycoco", default="True")
p.add_argument("--batch", type=int, default=16, help="dataloader batch（onnx 内部仍逐图跑）")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--device", default="cpu", help="decode/metric 所在设备（ort 固定 CPU EP）")
a = p.parse_args()
requested_e2e = a.end2end.lower() == "true"
rect = a.rect.lower() == "true"
pycoco = a.pycoco.lower() == "true"
device = select_device(a.device)

# float 参考模型：只为 head 的 _inference/postprocess 结构与 stride/names（不参与数值）
m = YOLO(a.model, task="detect").load(a.pretrained)
fm = m.model.float().to(device)
fm.model[-1].end2end = requested_e2e
e2e = bool(fm.model[-1].end2end)
if e2e != requested_e2e:
    print(
        f"[onnx] requested end2end={requested_e2e}, but {type(fm.model[-1]).__name__} has no one2one head; use end2end={e2e}",
        flush=True,
    )
hyp = dict(DEFAULT_CFG_DICT, **fm.args) if isinstance(fm.args, dict) else {}
hyp.setdefault("box", 7.5)
hyp.setdefault("cls", 0.5)
hyp.setdefault("dfl", 1.5)
fm.args = SimpleNamespace(**hyp)
fm.criterion = fm.init_criterion()
fm.eval()

head = fm.model[-1]
wrapper = OrtOne2One(
    a.onnx,
    device,
    box_channels=4 * int(getattr(head, "reg_max", 1)),
    score_channels=head.nc,
    end2end=head.end2end,
)
print(f"[ort] {a.onnx} 加载成功（CPU EP，batch 固定 1 逐图）", flush=True)

data_dict = check_det_dataset(a.data)
gs = max(int(fm.stride.max()), 32)
ns = argparse.Namespace(
    task="detect",
    data=a.data,
    imgsz=a.imgsz,
    batch=a.batch,
    workers=a.workers,
    fraction=1.0,
    augment=False,
    erasing=0.0,
    flipud=0.0,
    fliplr=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    mosaic=0.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    auto_augment=None,
    single_cls=False,
    classes=None,
    overlap_mask=False,
    mask_ratio=4,
    rect=rect,
    cache=False,
)
vds = build_yolo_dataset(ns, data_dict["val"], a.batch, data_dict, mode="val", rect=rect, stride=gs)
vl = build_dataloader(vds, batch=a.batch, workers=a.workers, shuffle=False, rank=-1, drop_last=False)

cfg = copy.deepcopy(DEFAULT_CFG_DICT)
cfg.update(
    {
        "task": "detect",
        "mode": "val",
        "data": a.data,
        "imgsz": a.imgsz,
        "batch": a.batch,
        "device": a.device,
        "workers": a.workers,
        "split": "val",
        "end2end": e2e,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "half": False,
        "plots": False,
        "save_json": pycoco,
        "save_hybrid": False,
    }
)
validator = DetectionValidator(dataloader=vl, args=cfg)
ft = FakeTrainer(fm, wrapper, data_dict, device, e2e)
results = validator(trainer=ft)
ul_map = results.get("metrics/mAP50-95(B)", 0.0)
ul_map50 = results.get("metrics/mAP50(B)", 0.0)

coco_map = None
if pycoco and getattr(validator, "jdict", None):
    validator.save_dir.mkdir(parents=True, exist_ok=True)
    pj = validator.save_dir / "predictions.json"
    with open(pj, "w") as f:
        json.dump(validator.jdict, f)
    print(f"[pycoco] {len(validator.jdict)} 条 → {pj}，eval_json ...", flush=True)
    try:
        st = validator.eval_json(dict(results))
        coco_map = st.get("metrics/mAP50-95(B)", None)
    except Exception:
        import traceback

        traceback.print_exc()

print("\n" + "=" * 60, flush=True)
print(f">>> ONNX {a.onnx}  rect={rect}", flush=True)
print(f"    ultralytics-metric  mAP50-95={ul_map:.4f}  mAP50={ul_map50:.4f}", flush=True)
if coco_map is not None:
    print(f"    pycocotools/COCO    mAP50-95={coco_map:.4f}", flush=True)
print("=" * 60, flush=True)
print("EVAL_ONNX_DONE", flush=True)
