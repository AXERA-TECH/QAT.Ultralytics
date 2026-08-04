#!/usr/bin/env python3
"""统一 float baseline eval（替代 eval_float_o2o.py / eval_float_o2o_v2.py）。.

float 模型走标准 model.val（head.forward 的 fuse+eval bug 已修：one2many 空守卫），
不需要 PT2E prepare。与 `eval.py qat` 使用相同口径参数，保证 QAT vs float 同口径对比。

口径（默认 = 官方对标口径 square + pycocotools）：
  --end2end True one2one / False one2many
  --rect     False square640（官方）/ True 矩形
  --pycoco   True  save_json=True 触发 pycocotools/COCO

用法：
  python eval.py float --weights yolo26n.pt --device 2
  python eval.py float --weights yolo26n.pt --end2end False --rect True
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")  # ordinal=nvidia-smi index
import argparse
import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO


def _bool(x):
    return str(x).lower() in ("true", "1", "yes")


def main():
    p = argparse.ArgumentParser(description="Unified float baseline eval")
    p.add_argument("--weights", default="yolo26n.pt", help="float 权重（.pt）")
    p.add_argument("--task", default="detect", help="detect/segment/obb/pose/classify")
    p.add_argument("--data", default="coco.yaml")
    p.add_argument("--end2end", default="True", type=_bool, help="True=one2one / False=one2many")
    p.add_argument("--rect", default="False", type=_bool, help="False=square640(官方) / True=矩形")
    p.add_argument("--pycoco", default="True", type=_bool, help="save_json→pycocotools/COCO")
    p.add_argument("--device", default="0", help="nvidia-smi 卡号（PCI order）")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=64)
    a = p.parse_args()

    if a.task == "classify":
        # Classification has no boxes/NMS/end2end; report Top-1/Top-5 instead of mAP.
        print(f"[eval float] classify  weights={a.weights}", flush=True)
        m = YOLO(a.weights, task="classify")
        r = m.val(data=a.data, imgsz=a.imgsz, batch=a.batch, device=a.device, half=False, plots=False, verbose=False)
        print("\n" + "=" * 60, flush=True)
        print(">>> FLOAT classify", flush=True)
        print(f"    ultralytics-metric  top1={r.top1:.4f}  top5={r.top5:.4f}", flush=True)
        print("=" * 60, flush=True)
        print("EVAL_FLOAT_DONE", flush=True)
        return

    tag = "one2one" if a.end2end else "one2many"
    print(f"[eval float] {tag}  rect={a.rect}  pycoco={a.pycoco}  weights={a.weights}", flush=True)

    m = YOLO(a.weights)
    r = m.val(
        data=a.data,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        end2end=a.end2end,
        conf=0.001,
        iou=0.7,
        rect=a.rect,
        save_json=a.pycoco,
        half=False,
        plots=False,
        verbose=False,
    )

    print("\n" + "=" * 60, flush=True)
    print(f">>> FLOAT {tag}  rect={a.rect}", flush=True)
    print(f"    ultralytics-metric  mAP50-95={r.box.map:.4f}  mAP50={r.box.map50:.4f}", flush=True)
    if getattr(r, "pose", None) is not None:
        print(f"    pose-metric         mAP50-95={r.pose.map:.4f}  mAP50={r.pose.map50:.4f}", flush=True)
    print(
        "    pycocotools/COCO    见上方 'Evaluating ... COCO' 段（save_json=True 自动触发）" if a.pycoco else "",
        flush=True,
    )
    print("=" * 60, flush=True)
    print("EVAL_FLOAT_DONE", flush=True)


if __name__ == "__main__":
    main()
