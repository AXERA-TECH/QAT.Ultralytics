# mypy: allow-untyped-defs
from __future__ import annotations

import dataclasses
import json
from typing import Any

import torch
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
)
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer

from ultralytics.utils.ax_quantizer_utils import (
    OP_TO_ANNOTATOR,
    QuantizationConfig,
    _convert_scalars_to_attrs,
    annotate_bias,
    propagate_annotation,
)

# from torch.fx import Node


__all__ = [
    "AXQuantizer",
    "get_quantization_config",
    "get_symmetric_quantization_config",
]


tmp_dtype_map = {
    "U4": torch.uint8,
    "S4": torch.int8,
    "U8": torch.uint8,
    "S8": torch.int8,
    "U16": torch.uint16,
    "S16": torch.int16,
}


@dataclasses.dataclass
class DtypeConf:
    dtype: torch.dtype = None
    qmin: int = None
    qmax: int = None


@dataclasses.dataclass
class QuantConf:
    input_dtype: DtypeConf = None
    weight_dtype: DtypeConf = None
    output_dtype: DtypeConf = None
    output_is_symmetric: bool | None = None
    act_observer: str | None = None
    output_act_observer: str | None = None
    weight_observer: str | None = None
    share_qparam: bool = False  # concat 输入输出共享量化参数（默认 False=当前行为）


@dataclasses.dataclass
class QuantizerRegionalConf:
    module_names: None
    module_type: None
    module_config: QuantizationConfig = None


def _build_act_fake_quant(
    observer_name: str,
    is_qat: bool,
    is_dynamic: bool,
) -> tuple[type[ObserverOrFakeQuantize], dict[str, Any]]:
    extra_args: dict[str, Any] = {"eps": 2**-12}
    if is_qat:
        if is_dynamic:
            extra_args["observer"] = MovingAverageMinMaxObserver.with_args(averaging_constant=1)
            return FakeQuantize, extra_args
        if observer_name in {"moving_avg", "default", "moving_average_minmax"}:
            extra_args["observer"] = MovingAverageMinMaxObserver
            return FusedMovingAvgObsFakeQuantize, extra_args
        if observer_name == "minmax":
            extra_args["observer"] = MinMaxObserver
            return FakeQuantize, extra_args
        if observer_name == "histogram":
            extra_args["observer"] = HistogramObserver
            return FakeQuantize, extra_args
        raise ValueError(f"Unsupported activation observer: {observer_name}")

    if is_dynamic:
        return PlaceholderObserver, extra_args

    # PTQ(is_qat=False): 直接用 observer 做校准（对齐上游，不包 FakeQuantize）
    return HistogramObserver, extra_args


def _build_weight_fake_quant(
    observer_name: str,
    is_qat: bool,
    ch_axis: int,
    fused: bool = True,
) -> tuple[type[ObserverOrFakeQuantize], dict[str, Any]]:
    extra_args: dict[str, Any] = {"eps": 2**-12}
    if is_qat:
        if observer_name in {"moving_avg_per_channel", "default"}:
            extra_args["observer"] = MovingAveragePerChannelMinMaxObserver.with_args(ch_axis=ch_axis)
            # fused_moving_avg_obs_fake_quant 的 per-channel 只支持 ch_axis==0；
            # 需要非0轴（convtranspose）时由调用方传 fused=False 改用普通 FakeQuantize（对齐上游）
            return (FusedMovingAvgObsFakeQuantize if fused else FakeQuantize), extra_args
        if observer_name in {"per_channel", "per_channel_minmax"}:
            extra_args["observer"] = PerChannelMinMaxObserver.with_args(ch_axis=ch_axis)
            return FakeQuantize, extra_args
        if observer_name in {"moving_avg", "moving_average_minmax"}:
            extra_args["observer"] = MovingAverageMinMaxObserver
            return FusedMovingAvgObsFakeQuantize, extra_args
        if observer_name == "minmax":
            extra_args["observer"] = MinMaxObserver
            return FakeQuantize, extra_args
        raise ValueError(f"Unsupported weight observer: {observer_name}")

    # PTQ(is_qat=False): 直接用 observer 做校准（对齐上游，不包 FakeQuantize；保留 ch_axis 与 QAT 一致）
    extra_args["ch_axis"] = ch_axis
    return PerChannelMinMaxObserver, extra_args


# @functools.lru_cache
def get_quantization_config(
    is_symmetric: bool = False, is_qat: bool = True, is_dynamic: bool = False, quant_config: QuantConf = None
):
    act_observer_name = (quant_config.act_observer or "moving_avg").lower()
    output_act_observer_name = (quant_config.output_act_observer or act_observer_name).lower()
    weight_observer_name = (quant_config.weight_observer or "moving_avg_per_channel").lower()

    act_qscheme = torch.per_tensor_symmetric if is_symmetric else torch.per_tensor_affine
    output_qscheme = (
        torch.per_tensor_symmetric
        if quant_config.output_is_symmetric is True
        else torch.per_tensor_affine
        if quant_config.output_is_symmetric is False
        else act_qscheme
    )
    act_observer_or_fake_quant_ctr, extra_args = _build_act_fake_quant(
        act_observer_name, is_qat=is_qat, is_dynamic=is_dynamic
    )
    output_observer_or_fake_quant_ctr, output_extra_args = _build_act_fake_quant(
        output_act_observer_name, is_qat=is_qat, is_dynamic=is_dynamic
    )

    input_dtype = quant_config.input_dtype
    act_spec_args = {
        "dtype": None,
        "quant_min": None,
        "quant_max": None,
        "qscheme": act_qscheme,
        "is_dynamic": is_dynamic,
        "observer_or_fake_quant_ctr": act_observer_or_fake_quant_ctr.with_args(**extra_args),
    }

    if input_dtype is not None:
        input_quantization_spec = QuantizationSpec(
            dtype=input_dtype.dtype,
            quant_min=input_dtype.qmin,
            quant_max=input_dtype.qmax,
            **{k: v for k, v in act_spec_args.items() if k not in ("dtype", "quant_min", "quant_max")},
        )
    else:
        input_quantization_spec = None

    # output
    output_dtype = quant_config.output_dtype
    if output_dtype is not None:
        output_quantization_spec = QuantizationSpec(
            dtype=output_dtype.dtype,
            quant_min=output_dtype.qmin,
            quant_max=output_dtype.qmax,
            **{
                **{k: v for k, v in act_spec_args.items() if k not in ("dtype", "quant_min", "quant_max", "qscheme")},
                "qscheme": output_qscheme,
                "observer_or_fake_quant_ctr": output_observer_or_fake_quant_ctr.with_args(**output_extra_args),
            },
        )
    else:
        output_quantization_spec = None

    # weight
    weight_qscheme = torch.per_channel_symmetric
    weight_observer_or_fake_quant_ctr, extra_args = _build_weight_fake_quant(
        weight_observer_name, is_qat=is_qat, ch_axis=0
    )
    weight_dtype = quant_config.weight_dtype
    if weight_dtype is not None:
        weight_quantization_spec = QuantizationSpec(
            dtype=weight_dtype.dtype,
            quant_min=weight_dtype.qmin,
            quant_max=weight_dtype.qmax,
            qscheme=weight_qscheme,
            ch_axis=0,
            is_dynamic=False,
            observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(**extra_args),
        )
    else:
        weight_quantization_spec = None

    # convtranspose weight —— 权重 layout=[in,out,kH,kW]，输出通道在轴=1；
    # fused_moving_avg_obs_fake_quant per-channel 只支持 axis==0，故显式 fused=False 用普通 FakeQuantize（对齐上游）
    weight_qscheme = torch.per_channel_symmetric
    weight_observer_or_fake_quant_ctr, extra_args = _build_weight_fake_quant(
        weight_observer_name, is_qat=is_qat, ch_axis=1, fused=False
    )

    weight_dtype = quant_config.weight_dtype
    if weight_dtype is not None:
        weight_trans_quantization_spec = QuantizationSpec(
            dtype=weight_dtype.dtype,
            quant_min=weight_dtype.qmin,
            quant_max=weight_dtype.qmax,
            qscheme=weight_qscheme,
            ch_axis=0,
            is_dynamic=False,
            observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(**extra_args),
        )
    else:
        weight_trans_quantization_spec = None

    # bias
    bias_quantization_spec = None

    if is_dynamic:
        quantization_config = QuantizationConfig(
            input_quantization_spec,
            None,
            weight_quantization_spec,
            weight_trans_quantization_spec,
            bias_quantization_spec,
            is_qat,
        )
    else:
        quantization_config = QuantizationConfig(
            input_quantization_spec,
            output_quantization_spec,
            weight_quantization_spec,
            weight_trans_quantization_spec,
            bias_quantization_spec,
            is_qat,
        )
    if getattr(quant_config, "share_qparam", False):
        try:
            quantization_config.share_qparam = True
        except Exception:
            object.__setattr__(quantization_config, "share_qparam", True)
    return quantization_config


def get_config(config: dict[str, Any], inherit_output_from_input: bool = True):
    is_symmetric = config["is_symmetric"]
    act_observer = config.get("act_observer", "moving_avg")
    output_act_observer = config.get("output_act_observer")
    weight_observer = config.get("weight_observer", "moving_avg_per_channel")
    if config["input"]["dtype"] == "FP32":
        quant_config = QuantConf(share_qparam=config.get("share_qparam", False))
    else:
        input_dtype = DtypeConf(
            dtype=tmp_dtype_map[config["input"]["dtype"]], qmin=config["input"]["qmin"], qmax=config["input"]["qmax"]
        )

        if "weight" in config:
            weight_dtype = DtypeConf(
                dtype=tmp_dtype_map[config["weight"]["dtype"]],
                qmin=config["weight"]["qmin"],
                qmax=config["weight"]["qmax"],
            )
        else:
            weight_dtype = None

        if "output" in config:  # 解耦：允许独立设 output dtype（如 silu input=U8 output=U16）
            output_dtype = DtypeConf(
                dtype=tmp_dtype_map[config["output"]["dtype"]],
                qmin=config["output"]["qmin"],
                qmax=config["output"]["qmax"],
            )
        elif inherit_output_from_input:
            output_dtype = input_dtype  # 向后兼容：无 output 字段 → output=input
        else:
            output_dtype = None  # regional 缺省 output：仅覆盖目标算子 input qspec
        quant_config = QuantConf(
            input_dtype=input_dtype,
            weight_dtype=weight_dtype,
            output_dtype=output_dtype,
            output_is_symmetric=config.get("output_is_symmetric"),
            act_observer=act_observer,
            output_act_observer=output_act_observer,
            weight_observer=weight_observer,
            share_qparam=config.get("share_qparam", False),
        )
    return is_symmetric, quant_config


def load_global_config(global_config: dict[str, str], is_qat: bool = True):
    is_symmetric, quant_config = get_config(global_config, inherit_output_from_input=True)
    global_quantization_config = get_quantization_config(
        is_symmetric=is_symmetric, is_qat=is_qat, quant_config=quant_config
    )
    return global_quantization_config


def load_regional_config(regional_config: dict[str, str], is_qat: bool = True):
    module_names = regional_config.get("module_names", None)
    module_type = regional_config["module_type"]
    raw_module_config = regional_config.get("module_config", None)
    if raw_module_config is None:
        module_config = None
    else:
        is_symmetric, quant_config = get_config(raw_module_config, inherit_output_from_input=False)
        module_config = get_quantization_config(is_symmetric=is_symmetric, is_qat=is_qat, quant_config=quant_config)
    regional_quantization_config = QuantizerRegionalConf(
        module_names=module_names, module_type=module_type, module_config=module_config
    )
    return regional_quantization_config


def ax_load_config(config_file: str, is_qat: bool = True):

    with open(config_file) as f:
        config = json.load(f)

    # global
    global_config = config["global_config"]
    global_quantization_config = load_global_config(global_config, is_qat=is_qat)

    # rregional
    regional_configs = config["regional_configs"]
    regional_quantization_configs = []
    for regional_config in regional_configs:
        regional_quantization_config = load_regional_config(regional_config, is_qat=is_qat)
        regional_quantization_configs.append(regional_quantization_config)

    return global_quantization_config, regional_quantization_configs


def remove_reused_bn_param_hack(model: torch.fx.GraphModule):
    for node in model.graph.nodes:
        if (
            node.target == torch.ops.aten.add_.Tensor
            and node.args[1] == 1
            and torch.nn.modules.batchnorm.BatchNorm2d in [val[1] for val in node.meta["source_fn_stack"]]
        ):
            last_node = node.args[0]
            if last_node.op != "get_attr":
                assert (
                    last_node.target == torch.ops.aten.add_.Tensor
                    and last_node.op == "call_function"
                    and last_node.args[0].op == "get_attr"
                    and last_node.args[1] == 1
                    and torch.nn.modules.batchnorm.BatchNorm2d in [val[1] for val in last_node.meta["source_fn_stack"]]
                )
                node.args = (last_node.args[0], node.args[1])


class AXQuantizer(Quantizer):
    STATIC_QAT_ONLY_OPS = [
        "conv_bn_relu",
        "conv_bn",
        "conv_transpose_bn_relu",
        "conv_transpose_bn",
    ]

    # static quantization ops (both PTQ and QAT)
    # Preserve the order that fusions come before singular ops
    STATIC_OPS = [
        "linear_relu",
        "linear",
        "conv_relu",
        "conv",
        "conv_transpose_relu",
        "adaptive_avg_pool2d",
        # TODO: move this to BoltNNQuantizer?
        "gru_io_only",
        "add_relu",
        "add",
        "mul_relu",
        "mul",
        "silu",
        "cat",
    ]

    DYNAMIC_OPS = [
        "linear",
    ]

    OPS = [
        "add",  # add, add_relu
        "avgpool2d",  # adaptive_avg_pool2d
        "concat",  # "cat"
        "conv",  # conv, conv_relu, conv_bn, conv_bn_relu
        "convtranspose",  # conv_transpose_relu, conv_transpose_bn, conv_transpose_bn_relu
        "layernorm",
        "linear",  # linear, linaer_relu
        "matmul",
        "mul",  # mul, mul_relu
        "gelu",
        "glu",
        "gridsample",
        "groupnorm",
        "silu",
        "softmax",
        "split",
    ]

    def __init__(self, annotate_bias: bool = True) -> None:
        super().__init__()
        self._annotate_bias = annotate_bias

        # init global
        self.global_config: QuantizationConfig | None = None
        # init regional
        self.regional_configs: list[QuantizerRegionalConf] = []
        self.init_regional()

    def init_regional(self):
        regional_matmul = {
            "module_names": None,
            "module_type": "matmul",
            "module_config": {
                "is_symmetric": True,
                "input": {"dtype": "S16", "qmin": -32767, "qmax": 32767},
            },
        }
        regional_matmul_config = load_regional_config(regional_matmul)
        self.regional_configs.append(regional_matmul_config)
        regional_gridsample = {
            "module_names": None,
            "module_type": "gridsample",
            "module_config": {
                "is_symmetric": True,
                "input": {"dtype": "S16", "qmin": -32767, "qmax": 32767},
            },
        }
        regional_gridsample_config = load_regional_config(regional_gridsample)
        self.regional_configs.append(regional_gridsample_config)
        return self

    def set_global(self, global_config: QuantizationConfig) -> AXQuantizer:
        self.global_config = global_config
        return self

    def set_regional(self, regional_configs: list[QuantizerRegionalConf]):
        self.regional_configs.extend(regional_configs)
        return self

    def transform_for_annotation(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """Transforms scalar values to tensor attributes."""
        return _convert_scalars_to_attrs(model)

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # global
        assert self.global_config is not None
        for op in self.OPS:
            OP_TO_ANNOTATOR[op](model, self.global_config, is_global=True)
        propagate_annotation(model, self.global_config)

        if self.regional_configs is not None:
            for regional_config in self.regional_configs:
                module_names = regional_config.module_names
                module_type = regional_config.module_type
                module_config = regional_config.module_config
                if module_type not in self.OPS:
                    continue
                OP_TO_ANNOTATOR[module_type](model, module_config, module_names, is_global=False)

        if self._annotate_bias:
            annotate_bias(model)

        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass
