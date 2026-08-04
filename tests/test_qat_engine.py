try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - allows manual execution in lean envs without pytest installed

    class _PytestStub:
        class mark:
            @staticmethod
            def slow(func):
                return func

    pytest = _PytestStub()

from ultralytics import YOLO


def _run_qat_train(model_name: str, data: str):
    """Run a minimal PT2E QAT training smoke test and return the training metrics object."""
    model = YOLO(model_name)
    return model.train(
        data=data,
        imgsz=32,
        epochs=1,
        batch=2,
        device="cpu",
        workers=0,
        save=False,
        plots=False,
        optimizer="SGD",
        lr0=1e-4,
        qat=True,
        qat_validate=True,
    )


@pytest.mark.slow
def test_detect_qat_validate_smoke():
    """Ensure YOLO26 detect PT2E QAT can train and validate for one epoch."""
    metrics = _run_qat_train("yolo26n.yaml", "coco8.yaml")
    assert type(metrics).__name__ == "DetMetrics"


@pytest.mark.slow
def test_segment_qat_validate_smoke():
    """Ensure YOLO26 segment PT2E QAT can train and validate for one epoch."""
    metrics = _run_qat_train("yolo26n-seg.yaml", "coco8-seg.yaml")
    assert type(metrics).__name__ == "SegmentMetrics"
