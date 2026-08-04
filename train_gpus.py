"""Experimental multi-GPU PT2E QAT training entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("ULTRALYTICS_SKIP_DATASET_HASH", "1")

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
PROFILES = {
    "accuracy": {
        "config": "config-qat/config_siluInU16_attnS8_clsU16.json",
        "name": "yolo26n-qat-accuracy-gpus",
    },
    "throughput": {
        "config": "config-qat/config_siluInU8_attnS8_clsU16.json",
        "name": "yolo26n-qat-throughput-gpus",
    },
}


def parse_devices(value: str) -> str:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if len(devices) < 2:
        raise argparse.ArgumentTypeError("--devices requires at least two GPU indices, for example 0,1")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("--devices contains duplicate GPU indices")
    if any(not item.isdigit() for item in devices):
        raise argparse.ArgumentTypeError("--devices accepts comma-separated non-negative GPU indices")
    return ",".join(devices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None, help="Optional YOLO26 delivery shortcut.")
    parser.add_argument("--quant-config", metavar="PATH", help="QAT JSON config. Takes precedence over --profile.")
    parser.add_argument("--task", choices=("detect",), default="detect", help="Only detect DDP has been validated.")
    parser.add_argument("--model", default="yolo26n.yaml")
    parser.add_argument("--pretrained", default="yolo26n.pt")
    parser.add_argument("--data", default="coco.yaml")
    parser.add_argument("--devices", type=parse_devices, default="0,1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64, help="Global batch size across all ranks.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers per rank.")
    parser.add_argument("--project", default=None, help="Defaults to runs/detect.")
    parser.add_argument("--name", default=None)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--qat-ema", action="store_true", help="Enable QAT EMA independently on every rank.")
    parser.add_argument("--qat-validate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--end2end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr0", type=float, default=2e-5)
    parser.add_argument("--lrf", type=float, default=0.1)
    return parser.parse_args()


def resolve_config_and_name(args: argparse.Namespace) -> tuple[Path, str]:
    """Return the explicit QAT config or the config selected by a delivery profile."""
    if args.quant_config:
        config = Path(args.quant_config)
        default_name = f"{Path(args.model).stem}-qat-gpus"
    else:
        if args.profile is None:
            raise ValueError("Specify --quant-config PATH or choose --profile accuracy|throughput.")
        profile = PROFILES[args.profile]
        config = Path(profile["config"])
        default_name = profile["name"]

    if not config.is_file():
        raise FileNotFoundError(f"Missing QAT config: {config}")
    return config, default_name


def resolve_project_dir(project: str | None, task: str) -> str:
    """Keep relative run directories inside this repository, not Ultralytics' global RUNS_DIR."""
    path = Path(project) if project else ROOT / "runs" / task
    return str(path if path.is_absolute() else ROOT / path)


def main() -> None:
    args = parse_args()
    if "LOCAL_RANK" in os.environ:
        raise RuntimeError("Run train_gpus.py directly; Ultralytics will launch torch.distributed workers.")

    world_size = len(args.devices.split(","))
    if args.batch < world_size or args.batch % world_size:
        raise ValueError(f"--batch must be a positive multiple of GPU count {world_size}, got {args.batch}")

    config, default_name = resolve_config_and_name(args)
    if not Path(args.pretrained).is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {args.pretrained}")

    model = YOLO(args.model, task=args.task)
    model.load(args.pretrained)
    model.train(
        data=args.data,
        batch=args.batch,
        epochs=args.epochs,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.devices,
        project=resolve_project_dir(args.project, args.task),
        name=args.name or default_name,
        exist_ok=args.exist_ok,
        qat=True,
        qat_config=str(config),
        qat_validate=args.qat_validate,
        qat_ema=args.qat_ema,
        end2end=args.end2end,
        save_period=args.save_period,
        fraction=args.fraction,
        lr0=args.lr0,
        lrf=args.lrf,
    )


if __name__ == "__main__":
    main()
