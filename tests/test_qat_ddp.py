from pathlib import Path
from types import SimpleNamespace

import torch

import ultralytics.engine.trainer as trainer_module
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import dist as dist_utils


class _DummyTrainer:
    hub_session = None
    model = "yolo26n.yaml"

    def __init__(self, qat: bool):
        self.args = SimpleNamespace(
            qat=qat,
            model="yolo26n.yaml",
            pretrained="yolo26n.pt",
            task="detect",
            device="0,1",
            save_dir="runs/detect/test",
        )


class _ParallelLike(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module


def test_generate_qat_ddp_file_reenters_yolo_train(monkeypatch, tmp_path):
    monkeypatch.setattr(dist_utils, "USER_CONFIG_DIR", tmp_path)
    path = Path(dist_utils.generate_ddp_file(_DummyTrainer(qat=True)))
    content = path.read_text()

    compile(content, str(path), "exec")
    assert "from ultralytics import YOLO" in content
    assert 'model_path = overrides.pop("model")' in content
    assert 'pretrained = overrides.pop("pretrained", None)' in content
    assert "model.load(pretrained)" in content
    assert "model.train(**overrides)" in content


def test_qat_checkpoint_unwraps_parallel_model(tmp_path):
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.epoch = 0
    trainer.best_fitness = 0.5
    trainer.fitness = 0.5
    trainer.model = torch.nn.Linear(2, 2)
    trainer.qat_model = _ParallelLike(torch.nn.Linear(2, 2))
    trainer.qat_ema = None
    trainer.ema = None
    trainer.optimizer = torch.optim.SGD(trainer.qat_model.parameters(), lr=0.1)
    trainer.scaler = torch.amp.GradScaler("cuda", enabled=False)
    trainer.args = SimpleNamespace(qat=True)
    trainer.metrics = {}
    trainer.wdir = tmp_path / "weights"
    trainer.last = trainer.wdir / "last.pt"
    trainer.best = trainer.wdir / "best.pt"
    trainer.save_period = -1
    trainer.read_results_csv = dict

    trainer.save_model()
    checkpoint = torch.load(trainer.last, map_location="cpu", weights_only=False)

    assert checkpoint["qat_model"]
    assert all(not name.startswith("module.") for name in checkpoint["qat_model"])


def test_qat_ddp_builds_full_validation_loader_only_on_rank0(monkeypatch):
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.batch_size = 64
    trainer.world_size = 2
    trainer.epochs = 1
    trainer.data = {"train": "train", "val": "val"}
    trainer.args = SimpleNamespace(task="detect", nbs=64, weight_decay=0.0005, optimizer="SGD", lr0=0.1, momentum=0.9)
    trainer.qat_model = torch.nn.Linear(2, 2)
    calls = []
    train_loader = SimpleNamespace(dataset=range(64))
    trainer.get_dataloader = lambda path, batch_size, rank, mode: (
        calls.append((path, batch_size, rank, mode)) or (train_loader if mode == "train" else mode)
    )
    trainer.build_optimizer = lambda **kwargs: torch.optim.SGD(trainer.qat_model.parameters(), lr=0.1)
    trainer._setup_scheduler = lambda: None

    monkeypatch.setattr(trainer_module, "RANK", 0)
    trainer._build_train_pipeline()
    assert trainer.test_loader == "val"
    assert calls[-1] == ("val", 32, -1, "val")

    calls.clear()
    monkeypatch.setattr(trainer_module, "RANK", 1)
    trainer._build_train_pipeline()
    assert trainer.test_loader is None
    assert calls == [("train", 32, trainer_module.LOCAL_RANK, "train")]


def test_qat_ddp_validation_runs_on_rank0_and_broadcasts(monkeypatch):
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(qat_validate=True)
    trainer.qat_model = torch.nn.Linear(2, 2)
    trainer.qat_ema = None
    trainer.ema = None
    trainer.world_size = 2
    trainer.loss = torch.tensor(1.0)
    trainer.best_fitness = 0.0
    seen_world_sizes = []
    trainer.validator = lambda current: (
        seen_world_sizes.append(current.world_size)
        or {
            "metrics/mAP50-95(B)": 0.4,
            "fitness": 0.4,
        }
    )
    broadcasts = []
    monkeypatch.setattr(trainer_module, "RANK", 0)
    monkeypatch.setattr(
        trainer_module.dist, "broadcast_object_list", lambda values, src: broadcasts.append((values, src))
    )

    metrics, fitness = trainer.validate()

    assert seen_world_sizes == [1]
    assert trainer.world_size == 2
    assert metrics == {"metrics/mAP50-95(B)": 0.4}
    assert fitness == 0.4
    assert broadcasts and broadcasts[0][1] == 0


def test_qat_ddp_validation_nonzero_rank_uses_broadcast_metrics(monkeypatch):
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(qat_validate=True)
    trainer.qat_model = torch.nn.Linear(2, 2)
    trainer.qat_ema = None
    trainer.ema = None
    trainer.world_size = 2
    trainer.loss = torch.tensor(1.0)
    trainer.best_fitness = 0.0
    trainer.validator = lambda _: (_ for _ in ()).throw(AssertionError("nonzero rank must not validate"))

    def broadcast(values, src):
        values[0] = {"metrics/mAP50-95(B)": 0.4, "fitness": 0.4}

    monkeypatch.setattr(trainer_module, "RANK", 1)
    monkeypatch.setattr(trainer_module.dist, "broadcast_object_list", broadcast)

    metrics, fitness = trainer.validate()

    assert metrics == {"metrics/mAP50-95(B)": 0.4}
    assert fitness == 0.4


def test_rank0_only_validation_skips_distributed_stats_gather(monkeypatch):
    validator = DetectionValidator.__new__(DetectionValidator)
    validator.distributed_validation = False
    validator.metrics = SimpleNamespace(stats={"tp": []})
    validator.jdict = []
    monkeypatch.setattr(
        "ultralytics.models.yolo.detect.val.dist.gather_object",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("gather must be skipped")),
    )

    validator.gather_stats()
