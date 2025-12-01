# mypy: allow-untyped-defs
from __future__ import annotations

import copy
import json
import functools
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

import torch
import torch._dynamo as torchdynamo
import torch.nn.functional as F
from torch import Tensor
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
)
from torch.ao.quantization import observer, ObserverOrFakeQuantize
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer, DerivedQuantizationSpec
from torch.ao.quantization.quantizer.utils import _get_module_name_filter
from ultralytics.utils.ax_quantizer_utils import (
    _convert_scalars_to_attrs,
    OP_TO_ANNOTATOR,
    OperatorConfig,
    OperatorPatternType,
    propagate_annotation,
    QuantizationConfig,
    annotate_bias,
)
from torch.fx import Node


if TYPE_CHECKING:
    from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
    # from torch.fx import Node


__all__ = [
    "AXQuantizer",
    "get_symmetric_quantization_config",
    "get_quantization_config",
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


@dataclasses.dataclass
class QuantizerRegionalConf:
    module_names: None
    module_type: None
    module_config: QuantizationConfig = None


# @functools.lru_cache
def get_quantization_config(
    is_symmetric: bool = False,
    is_qat: bool = True,
    is_dynamic: bool = False,
    quant_config: QuantConf = None
):
    # input
    act_qscheme = (
        torch.per_tensor_symmetric if is_symmetric else torch.per_tensor_affine
    )
    extra_args: Dict[str, Any] = {"eps": 2**-12}

    if is_qat:
        if is_dynamic:
            act_observer_or_fake_quant_ctr = FakeQuantize
            dynamic_quant_observer = MovingAverageMinMaxObserver.with_args(
                averaging_constant=1
            )
            extra_args["observer"] = dynamic_quant_observer
        else:
            act_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize  # type: ignore[assignment]
    else:
        if is_dynamic:
            act_observer_or_fake_quant_ctr = PlaceholderObserver  # type: ignore[assignment]
        else:
            extra_args["observer"] =  HistogramObserver
            act_observer_or_fake_quant_ctr = FakeQuantize  # type: ignore[assignment]

    input_dtype = quant_config.input_dtype
    if input_dtype is not None:
        input_quantization_spec = QuantizationSpec(
            dtype=input_dtype.dtype,
            quant_min=input_dtype.qmin,
            quant_max=input_dtype.qmax,
            qscheme=act_qscheme,
            is_dynamic=is_dynamic,
            observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
                **extra_args,
            ),
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
            qscheme=act_qscheme,
            is_dynamic=is_dynamic,
            observer_or_fake_quant_ctr=act_observer_or_fake_quant_ctr.with_args(
                **extra_args,
            ),
        )
    else:
        output_quantization_spec = None

    # weight
    weight_qscheme = torch.per_channel_symmetric
    extra_args: Dict[str, Any] = {"eps": 2**-12}
    if is_qat:
        if weight_qscheme == torch.per_tensor_symmetric:
            extra_args["observer"] = MovingAverageMinMaxObserver
        else:
            extra_args["observer"] = MovingAveragePerChannelMinMaxObserver  # type: ignore[dict-item]
    
    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        MinMaxObserver
    )
    if is_qat:
        # TODO: qat + per channel?
        weight_observer_or_fake_quant_ctr = FusedMovingAvgObsFakeQuantize
    else:
        extra_args["observer"] = PerChannelMinMaxObserver
        weight_observer_or_fake_quant_ctr = FakeQuantize
    weight_dtype = quant_config.weight_dtype
    if weight_dtype is not None:
        weight_quantization_spec = QuantizationSpec(
            dtype=weight_dtype.dtype,
            quant_min=weight_dtype.qmin,
            quant_max=weight_dtype.qmax,
            qscheme=weight_qscheme,
            ch_axis=0,
            is_dynamic=False,
            observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
                **extra_args
            ),
        )
    else:
        weight_quantization_spec = None

    # convtranspose weight
    weight_qscheme = torch.per_channel_symmetric
    extra_args: Dict[str, Any] = {"eps": 2**-12}
    if is_qat:
        if weight_qscheme == torch.per_tensor_symmetric:
            extra_args["observer"] = MovingAverageMinMaxObserver
        else:
            extra_args["observer"] = MovingAveragePerChannelMinMaxObserver.with_args(ch_axis=1)  # type: ignore[dict-item]
    
    weight_observer_or_fake_quant_ctr: _ObserverOrFakeQuantizeConstructor = (
        MinMaxObserver
    )
    if is_qat:
        # TODO: qat + per channel?
        weight_observer_or_fake_quant_ctr = FakeQuantize
    else:
        weight_observer_or_fake_quant_ctr = PerChannelMinMaxObserver

    weight_dtype = quant_config.weight_dtype
    if weight_dtype is not None:
        weight_trans_quantization_spec = QuantizationSpec(
            dtype=weight_dtype.dtype,
            quant_min=weight_dtype.qmin,
            quant_max=weight_dtype.qmax,
            qscheme=weight_qscheme,
            ch_axis=0,
            is_dynamic=False,
            observer_or_fake_quant_ctr=weight_observer_or_fake_quant_ctr.with_args(
                **extra_args
            ),
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
    return quantization_config



def get_config(config: Dict[str, Any]):
    is_symmetric = config["is_symmetric"]
    if config["input"]["dtype"] == "FP32":
        quant_config = QuantConf()
    else:
        input_dtype = DtypeConf(
            dtype=tmp_dtype_map[config["input"]["dtype"]],
            qmin=config["input"]["qmin"],
            qmax=config["input"]["qmax"]
        )

        if "weight" in config:
            weight_dtype = DtypeConf(
                dtype=tmp_dtype_map[config["weight"]["dtype"]],
                qmin=config["weight"]["qmin"],
                qmax=config["weight"]["qmax"]
            )
        else:
            weight_dtype = None

        quant_config = QuantConf(
            input_dtype=input_dtype,
            weight_dtype=weight_dtype,
            output_dtype=input_dtype
        )
    return is_symmetric, quant_config


def load_global_config(global_config: Dict[str, str], is_qat: bool = True):
    is_symmetric, quant_config = get_config(global_config)
    global_quantization_config = get_quantization_config(is_symmetric=is_symmetric, is_qat=is_qat, quant_config=quant_config)
    return global_quantization_config


def load_regional_config(regional_config: Dict[str, str], is_qat: bool = True):
    module_names = regional_config.get("module_names", None)
    module_type = regional_config["module_type"]
    is_symmetric, quant_config = get_config(regional_config["module_config"])
    module_config = get_quantization_config(is_symmetric=is_symmetric, is_qat=is_qat, quant_config=quant_config)
    regional_quantization_config = QuantizerRegionalConf(
        module_names=module_names,
        module_type=module_type,
        module_config=module_config
    )
    return regional_quantization_config


def load_config(config_file: str, is_qat: bool = True):

    with open(config_file, 'r') as f:
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
            and torch.nn.modules.batchnorm.BatchNorm2d
            in [val[1] for val in node.meta["source_fn_stack"]]
        ):
            last_node = node.args[0]
            if last_node.op != "get_attr":
                assert (
                    last_node.target == torch.ops.aten.add_.Tensor
                    and last_node.op == "call_function"
                    and last_node.args[0].op == "get_attr"
                    and last_node.args[1] == 1
                    and torch.nn.modules.batchnorm.BatchNorm2d
                    in [val[1] for val in last_node.meta["source_fn_stack"]]
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
        self.global_config: Optional[QuantizationConfig] = None
        # init regional
        self.regional_configs: List[QuantizerRegionalConf] = []
        self.init_regional()

    def init_regional(self):
        # matlul
        regional_matmul = {
            "module_names": None,
            "module_type": "matmul",
            "module_config": {
                "is_symmetric": True,
                "input": {
                    "dtype": "S16",
                    "qmin": -32767,
                    "qmax": 32767
                },
            }
        }
        regional_matmul_config = load_regional_config(regional_matmul)
        self.regional_configs.append(regional_matmul_config)
        # gridsample
        regional_gridsample = {
            "module_names": None,
            "module_type": "gridsample",
            "module_config": {
                "is_symmetric": True,
                "input": {
                    "dtype": "S16",
                    "qmin": -32767,
                    "qmax": 32767
                },
            }
        }
        regional_gridsample_config = load_regional_config(regional_gridsample)
        self.regional_configs.append(regional_gridsample_config)
        return self
        
    
    def set_global(self, global_config: QuantizationConfig) -> AXQuantizer:
        self.global_config = global_config
        return self

    def set_regional(self, regional_configs: List[QuantizerRegionalConf]):
        self.regional_configs.extend(regional_configs)
        return self

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """Transforms scalar values to tensor attributes"""
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