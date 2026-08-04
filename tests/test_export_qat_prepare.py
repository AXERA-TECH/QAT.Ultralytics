from unittest.mock import Mock

import torch

import export as export_module


class _DummyYOLO:
    def __init__(self, *_args, **_kwargs):
        self.model = torch.nn.Identity()

    def load(self, _path):
        return self


class _PreparedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loaded_state = None
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return x

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state = (state_dict, strict)


def test_export_reuses_training_prepare_path(monkeypatch, tmp_path):
    weights_path = tmp_path / "best.pt"
    weights_path.touch()
    config_path = tmp_path / "qat.json"
    config_path.write_text("{}")
    qat_state = {"activation_post_process_0.scale": torch.tensor([0.5])}
    checkpoint = {
        "train_args": {
            "qat_config": str(config_path),
            "qat_dynamic_batch_max": 64,
            "qat_lsq": True,
        },
        "qat_model": qat_state,
    }
    prepared = _PreparedModel()
    prepare_mock = Mock(return_value=(torch.nn.Identity(), prepared))

    monkeypatch.setattr(export_module, "YOLO", _DummyYOLO)
    monkeypatch.setattr(export_module.torch, "load", Mock(return_value=checkpoint))
    monkeypatch.setattr(export_module, "prepare_pt2e_qat_model", prepare_mock)
    monkeypatch.setattr(export_module, "convert_pt2e", lambda model: model)

    cfg = export_module.ExportDefaults(
        task="detect",
        model="yolo26n.yaml",
        pretrained="yolo26n.pt",
        qat_weights=str(weights_path),
        out=str(tmp_path / "model.onnx"),
        qat_state_out=str(tmp_path / "state.pth"),
    )
    quantized, inputs = export_module.build_quantized_model(
        cfg,
        quant_config="unused.json",
        device="cpu",
        qat_onnx_imgsz=[1, 3, 640, 512],
    )

    assert quantized is prepared
    assert inputs.shape == (1, 3, 640, 512)
    assert prepared.loaded_state == (qat_state, True)
    prepare_mock.assert_called_once()
    prepare_kwargs = prepare_mock.call_args.kwargs
    assert isinstance(prepare_kwargs["float_model"], torch.nn.Identity)
    assert prepare_kwargs["float_model"].training is True
    assert prepare_kwargs["device"] == "cpu"
    assert prepare_kwargs["config_path"] == str(config_path)
    assert prepare_kwargs["imgsz"] == (640, 512)
    assert prepare_kwargs["dynamic_batch_max"] == 64
    assert prepare_kwargs["use_lsq"] is True


def test_export_initializes_observers_without_checkpoint(monkeypatch, tmp_path):
    prepared = _PreparedModel()
    prepare_mock = Mock(return_value=(torch.nn.Identity(), prepared))

    monkeypatch.setattr(export_module, "YOLO", _DummyYOLO)
    monkeypatch.setattr(export_module, "prepare_pt2e_qat_model", prepare_mock)
    monkeypatch.setattr(export_module, "convert_pt2e", lambda model: model)

    cfg = export_module.ExportDefaults(
        task="detect",
        model="yolo26n.yaml",
        pretrained="yolo26n.pt",
        qat_weights=str(tmp_path / "missing.pt"),
        out=str(tmp_path / "model.onnx"),
        qat_state_out=str(tmp_path / "state.pth"),
    )
    quantized, _ = export_module.build_quantized_model(
        cfg,
        quant_config="config-qat/config.json",
        device="cpu",
        qat_onnx_imgsz=[1, 3, 32, 32],
    )

    assert quantized is prepared
    assert prepared.loaded_state is None
    assert prepared.forward_calls == 1
