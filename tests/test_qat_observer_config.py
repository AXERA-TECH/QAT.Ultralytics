import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.ao.quantization.fake_quantize import FakeQuantize, FusedMovingAvgObsFakeQuantize
from torch.ao.quantization.observer import HistogramObserver, MovingAverageMinMaxObserver
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_qat_pt2e

from ultralytics.utils.ax_quantizer import (
    AXQuantizer,
    get_config,
    get_quantization_config,
    load_regional_config,
)
from ultralytics.utils.ax_quantizer_utils import get_input_act_qspec, get_output_act_qspec
from ultralytics.utils.qat_utils import resolve_qat_config_path


def _make_config(output_act_observer=None):
    config = {
        "is_symmetric": True,
        "output_is_symmetric": False,
        "act_observer": "moving_avg",
        "input": {"dtype": "S16", "qmin": -32767, "qmax": 32767},
        "output": {"dtype": "U8", "qmin": 0, "qmax": 255},
    }
    if output_act_observer is not None:
        config["output_act_observer"] = output_act_observer
    return config


def test_output_observer_can_differ_from_input_observer():
    is_symmetric, quant_config = get_config(_make_config(output_act_observer="histogram"))
    config = get_quantization_config(is_symmetric=is_symmetric, quant_config=quant_config)

    input_qspec = get_input_act_qspec(config)
    output_qspec = get_output_act_qspec(config)
    input_fake_quant = input_qspec.observer_or_fake_quant_ctr()
    output_fake_quant = output_qspec.observer_or_fake_quant_ctr()

    assert isinstance(input_fake_quant, FusedMovingAvgObsFakeQuantize)
    assert isinstance(input_fake_quant.activation_post_process, MovingAverageMinMaxObserver)
    assert input_qspec.dtype == torch.int16
    assert isinstance(output_fake_quant, FakeQuantize)
    assert isinstance(output_fake_quant.activation_post_process, HistogramObserver)
    assert output_qspec.dtype == torch.uint8


def test_output_observer_defaults_to_input_observer():
    is_symmetric, quant_config = get_config(_make_config())
    config = get_quantization_config(is_symmetric=is_symmetric, quant_config=quant_config)

    input_fake_quant = get_input_act_qspec(config).observer_or_fake_quant_ctr()
    output_fake_quant = get_output_act_qspec(config).observer_or_fake_quant_ctr()

    assert isinstance(input_fake_quant, FusedMovingAvgObsFakeQuantize)
    assert isinstance(output_fake_quant, FusedMovingAvgObsFakeQuantize)


def test_regional_config_without_explicit_output_has_no_output_qspec():
    regional = load_regional_config(
        {
            "module_names": ["conv2d_117"],
            "module_type": "conv",
            "module_config": {
                "is_symmetric": False,
                "input": {"dtype": "U16", "qmin": 0, "qmax": 65535},
                "weight": {"dtype": "S8", "qmin": -127, "qmax": 127},
            },
        }
    )

    assert get_input_act_qspec(regional.module_config).dtype == torch.uint16
    assert get_output_act_qspec(regional.module_config) is None


def test_accuracy_attention_s8_delivery_config_contract():
    config_path = Path(__file__).parents[1] / "config-qat/config_siluInU16_attnS8_clsU16.json"
    config = json.loads(config_path.read_text())
    regional = config["regional_configs"]

    assert config["global_config"]["input"] == {"dtype": "U8", "qmin": 0, "qmax": 255}
    assert config["global_config"]["weight"] == {"dtype": "S8", "qmin": -127, "qmax": 127}
    silu = next(item for item in regional if item["module_type"] == "silu")
    assert silu["module_names"] is None
    assert silu["module_config"]["input"] == {"dtype": "U16", "qmin": 0, "qmax": 65535}
    assert silu["module_config"]["output"] == {"dtype": "U8", "qmin": 0, "qmax": 255}

    shared_head_silu = next(
        item for item in regional if item["module_type"] == "silu" and item["module_names"] == ["silu__68"]
    )
    assert shared_head_silu["module_config"]["output"] == {"dtype": "U16", "qmin": 0, "qmax": 65535}

    tower_silu = next(
        item
        for item in regional
        if item["module_type"] == "silu" and item["module_names"] == [f"silu__{index}" for index in range(97, 105)]
    )
    assert tower_silu["module_config"]["input"] == {"dtype": "U16", "qmin": 0, "qmax": 65535}
    assert tower_silu["module_config"]["output"] == {"dtype": "U16", "qmin": 0, "qmax": 65535}

    tower_conv = next(
        item
        for item in regional
        if item["module_type"] == "conv"
        and item["module_names"]
        == [
            "conv2d_117",
            "conv2d_118",
            "conv2d_119",
            "conv2d_121",
            "conv2d_122",
            "conv2d_123",
            "conv2d_124",
        ]
    )
    assert tower_conv["module_config"]["input"] == {"dtype": "U16", "qmin": 0, "qmax": 65535}
    assert "output" not in tower_conv["module_config"]

    output_conv = next(
        item
        for item in regional
        if item["module_type"] == "conv" and item["module_names"] == ["conv2d_120", "conv2d_125"]
    )
    assert output_conv["module_config"]["input"] == {"dtype": "U16", "qmin": 0, "qmax": 65535}
    assert output_conv["module_config"]["output"] == {"dtype": "U16", "qmin": 0, "qmax": 65535}

    first_matmul = next(
        item for item in regional if item["module_type"] == "matmul" and item["module_names"] == ["matmul", "matmul_2"]
    )
    assert first_matmul["module_config"]["output"] == {"dtype": "S8", "qmin": -127, "qmax": 127}
    attention_mul = next(item for item in regional if item["module_type"] == "mul")
    assert attention_mul["module_names"] == ["mul_75", "mul_152"]
    assert attention_mul["module_config"]["output"] == {"dtype": "S8", "qmin": -127, "qmax": 127}

    shared_head_conv = next(
        item for item in regional if item["module_type"] == "conv" and item["module_names"] == ["conv2d_108"]
    )
    assert shared_head_conv["module_config"]["input"] == {"dtype": "U16", "qmin": 0, "qmax": 65535}
    assert "output" not in shared_head_conv["module_config"]

    qkv_conv = next(
        item
        for item in regional
        if item["module_type"] == "conv" and item["module_names"] == ["conv2d_34", "conv2d_72"]
    )
    pe_conv = next(
        item
        for item in regional
        if item["module_type"] == "conv" and item["module_names"] == ["conv2d_35", "conv2d_73"]
    )
    assert qkv_conv["module_config"]["output"] == {"dtype": "S8", "qmin": -127, "qmax": 127}
    assert pe_conv["module_config"]["input"] == {"dtype": "S8", "qmin": -127, "qmax": 127}


def test_throughput_only_changes_global_silu_input_to_u8():
    root = Path(__file__).parents[1]
    accuracy = json.loads((root / "config-qat/config_siluInU16_attnS8_clsU16.json").read_text())
    throughput = json.loads((root / "config-qat/config_siluInU8_attnS8_clsU16.json").read_text())

    accuracy_global_silu = next(
        item for item in accuracy["regional_configs"] if item["module_type"] == "silu" and item["module_names"] is None
    )
    throughput_global_silu = next(
        item
        for item in throughput["regional_configs"]
        if item["module_type"] == "silu" and item["module_names"] is None
    )
    assert accuracy_global_silu["module_config"]["input"] == {"dtype": "U16", "qmin": 0, "qmax": 65535}
    assert throughput_global_silu["module_config"]["input"] == {"dtype": "U8", "qmin": 0, "qmax": 255}
    assert throughput_global_silu["module_config"]["output"] == {"dtype": "U8", "qmin": 0, "qmax": 255}

    normalized_throughput = copy.deepcopy(throughput)
    normalized_global_silu = next(
        item
        for item in normalized_throughput["regional_configs"]
        if item["module_type"] == "silu" and item["module_names"] is None
    )
    normalized_global_silu["module_config"]["input"] = copy.deepcopy(accuracy_global_silu["module_config"]["input"])
    assert normalized_throughput == accuracy


@pytest.mark.parametrize(
    ("base_name", "combined_name"),
    [
        ("config_siluInU16_attnS8.json", "config_siluInU16_attnS8_clsU16.json"),
        ("config_siluInU8_attnS8.json", "config_siluInU8_attnS8_clsU16.json"),
    ],
)
def test_attention_base_config_only_omits_optional_cls_u16_entries(base_name, combined_name):
    root = Path(__file__).parents[1] / "config-qat"
    base = json.loads((root / base_name).read_text())
    combined = json.loads((root / combined_name).read_text())

    def is_cls_u16_entry(item):
        module_config = item.get("module_config", {})
        input_dtype = module_config.get("input", {}).get("dtype")
        output_dtype = module_config.get("output", {}).get("dtype")
        if item.get("module_names") is None:
            return False
        return (item["module_type"] == "silu" and input_dtype == output_dtype == "U16") or (
            item["module_type"] == "conv" and input_dtype == "U16"
        )

    expected = copy.deepcopy(combined)
    expected["regional_configs"] = [item for item in expected["regional_configs"] if not is_cls_u16_entry(item)]
    assert base == expected


@pytest.mark.parametrize(
    ("legacy_name", "current_name"),
    [
        ("config_exp57_attn_s8.json", "config_siluInU16_attnS8_clsU16.json"),
        ("config_exp58_silu_u8_attn_s8.json", "config_siluInU8_attnS8_clsU16.json"),
    ],
)
def test_legacy_checkpoint_config_name_resolves_to_delivery_config(monkeypatch, legacy_name, current_name):
    root = Path(__file__).parents[1]
    monkeypatch.chdir(root)

    resolved = resolve_qat_config_path(root / "config-qat" / legacy_name)

    assert resolved == root / "config-qat" / current_name


def test_regional_matmul_explicit_output_qspec_is_applied():
    class MatMulModel(nn.Module):
        def forward(self, x, y):
            return torch.matmul(x, y)

    is_symmetric, global_quant_conf = get_config(
        {
            "is_symmetric": False,
            "input": {"dtype": "U8", "qmin": 0, "qmax": 255},
        }
    )
    global_config = get_quantization_config(is_symmetric=is_symmetric, quant_config=global_quant_conf)
    regional_config = load_regional_config(
        {
            "module_names": ["matmul"],
            "module_type": "matmul",
            "module_config": {
                "is_symmetric": True,
                "output_is_symmetric": True,
                "input": {"dtype": "S8", "qmin": -127, "qmax": 127},
                "output": {"dtype": "S8", "qmin": -127, "qmax": 127},
            },
        }
    )
    model = MatMulModel().train()
    exported = torch.export.export_for_training(
        model,
        (torch.randn(2, 4, 8), torch.randn(2, 8, 4)),
    ).module()
    quantizer = AXQuantizer().set_global(global_config).set_regional([regional_config])
    annotated = quantizer.annotate(quantizer.transform_for_annotation(exported))
    matmul = next(node for node in annotated.graph.nodes if node.target == torch.ops.aten.matmul.default)

    annotation = matmul.meta["quantization_annotation"]
    assert annotation.output_qspec.dtype == torch.int8
    assert all(qspec.dtype == torch.int8 for qspec in annotation.input_qspec_map.values())


class _ConvPatternModel(nn.Module):
    def __init__(self, use_bn: bool, use_relu: bool):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1, bias=not use_bn)
        self.bn = nn.BatchNorm2d(4, eps=1e-3, momentum=0.03) if use_bn else nn.Identity()
        self.act = nn.ReLU() if use_relu else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def _make_regional_conv_output_quantizer():
    is_symmetric, global_quant_conf = get_config(
        {
            "is_symmetric": False,
            "input": {"dtype": "U8", "qmin": 0, "qmax": 255},
            "weight": {"dtype": "S8", "qmin": -127, "qmax": 127},
        }
    )
    global_config = get_quantization_config(is_symmetric=is_symmetric, quant_config=global_quant_conf)
    regional_config = load_regional_config(
        {
            "module_names": None,
            "module_type": "conv",
            "module_config": {
                "is_symmetric": False,
                "output_is_symmetric": True,
                "input": {"dtype": "U8", "qmin": 0, "qmax": 255},
                "output": {"dtype": "S16", "qmin": -32767, "qmax": 32767},
                "weight": {"dtype": "S8", "qmin": -127, "qmax": 127},
            },
        }
    )
    return AXQuantizer().set_global(global_config).set_regional([regional_config])


@pytest.mark.parametrize(
    ("use_bn", "use_relu", "expected_output_target"),
    [
        (False, False, torch.ops.aten.conv2d.default),
        (True, False, torch.ops.aten.batch_norm.default),
        (True, True, torch.ops.aten.relu.default),
    ],
)
def test_regional_conv_output_uses_canonical_fused_pattern_output(use_bn, use_relu, expected_output_target):
    model = _ConvPatternModel(use_bn=use_bn, use_relu=use_relu).train()
    exported = torch.export.export_for_training(model, (torch.randn(2, 4, 8, 8),)).module()
    quantizer = _make_regional_conv_output_quantizer()
    annotated = quantizer.annotate(quantizer.transform_for_annotation(exported))

    conv_node = next(
        node
        for node in annotated.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.conv2d.default
    )
    pattern_output = conv_node.meta["_ax_quantization_pattern_output"]
    output_qspec = pattern_output.meta["quantization_annotation"].output_qspec

    assert pattern_output.target == expected_output_target
    assert output_qspec.dtype == torch.int16
    if use_bn:
        assert pattern_output is not conv_node
        assert conv_node.meta["quantization_annotation"].output_qspec is None


def test_regional_conv_bn_output_qspec_preserves_bn_folding():
    model = _ConvPatternModel(use_bn=True, use_relu=False).train()
    exported = torch.export.export_for_training(model, (torch.randn(2, 4, 8, 8),)).module()
    prepared = prepare_qat_pt2e(exported, _make_regional_conv_output_quantizer())
    converted = convert_pt2e(prepared)

    assert not any("batch_norm" in str(node.target) for node in converted.graph.nodes)
