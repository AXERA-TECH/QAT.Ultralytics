# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional

import operator
import torch
import torch.nn.functional as F
from torch import Tensor
from torch._subclasses import FakeTensor
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix
from torch.ao.quantization.pt2e.export_utils import _WrapperModule
from torch.ao.quantization.pt2e.utils import (
    _conv1d_bn_example_inputs,
    _conv2d_bn_example_inputs,
    _get_aten_graph_module_for_pattern,
    _is_conv_node,
    _is_conv_transpose_node,
)
from torch.ao.quantization import observer, ObserverOrFakeQuantize, FakeQuantizeBase
from torch.ao.quantization.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    QuantizationSpecBase,
    SharedQuantizationSpec,
    DerivedQuantizationSpec,
)
from torch.ao.quantization.quantizer.utils import (
    _annotate_input_qspec_map,
    _annotate_output_qspec,
)
from torch.fx import Node
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
    InternalMatch,
)
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
from torch.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize 



__all__ = [
    "OperatorConfig",
    "OperatorPatternType",
    "QuantizationConfig",
    "get_input_act_qspec",
    "get_output_act_qspec",
    "get_weight_qspec",
    "get_bias_qspec",
    "OP_TO_ANNOTATOR",
    "propagate_annotation",
]


# In the absence of better name, just winging it with QuantizationConfig
@dataclass(eq=True, frozen=True)
class QuantizationConfig:
    input_activation: Optional[QuantizationSpec]
    output_activation: Optional[QuantizationSpec]
    weight: Optional[QuantizationSpec]
    weight_trans: Optional[QuantizationSpec]
    bias: Optional[QuantizationSpec]
    # TODO: remove, since we can use observer_or_fake_quant_ctr to express this
    is_qat: bool = False


OperatorPatternType = List[Callable]
OperatorPatternType.__module__ = (
    "torch.ao.quantization.quantizer.xnnpack_quantizer_utils"
)

AnnotatorType = Callable[
    [
        torch.fx.GraphModule,
        Optional[QuantizationConfig],
        Optional[Callable[[Node], bool]],
    ],
    Optional[List[List[Node]]],
]
OP_TO_ANNOTATOR: Dict[str, AnnotatorType] = {}


def register_annotator(op: str):
    def decorator(annotator: AnnotatorType):
        OP_TO_ANNOTATOR[op] = annotator

    return decorator


class OperatorConfig(NamedTuple):
    # fix List[str] with List[List[Union[nn.Module, FunctionType, BuiltinFunctionType]]]
    # Basically we are mapping a quantization config to some list of patterns.
    # a pattern is defined as a list of nn module, function or builtin function names
    # e.g. [nn.Conv2d, torch.relu, torch.add]
    # We have not resolved whether fusion can be considered internal details of the
    # quantizer hence it does not need communication to user.
    # Note this pattern is not really informative since it does not really
    # tell us the graph structure resulting from the list of ops.
    config: QuantizationConfig
    operators: List[OperatorPatternType]


def _is_annotated(nodes: List[Node]):
    """
    Given a list of nodes (that represents an operator pattern),
    check if any of the node is annotated, return True if any of the node
    is annotated, otherwise return False
    """
    annotated = False
    for node in nodes:
        annotated = annotated or (
            "quantization_annotation" in node.meta
            and node.meta["quantization_annotation"]._annotated
        )
    return annotated


def _quant_spec_equal(qspec1: QuantizationSpec, qspec2: QuantizationSpec):
    # ignore observer_or_fake_quant_ctr
    return (
        qspec1.dtype == qspec2.dtype
        and qspec1.quant_min == qspec2.quant_min
        and qspec1.quant_max == qspec2.quant_max
        and qspec1.qscheme == qspec2.qscheme
        and qspec1.ch_axis == qspec2.ch_axis
        and qspec1.is_dynamic == qspec2.is_dynamic
    )


def _all_users_annotate_equal(input_node: Node, input_qspec: QuantizationSpec):
    if input_qspec is None or isinstance(input_qspec, SharedQuantizationSpec):
            return False

    for node in input_node.users:
        other_qspec = node.meta["quantization_annotation"].input_qspec_map[input_node] if "quantization_annotation" in node.meta else None
        if other_qspec is None or isinstance(other_qspec, SharedQuantizationSpec):
            continue
        if not _quant_spec_equal(other_qspec, input_qspec):
            return False
    return True


def _update_last_node_output_qspec(last_node: Node, node: Node, output_qspec: QuantizationSpec):
    input_qspec = node.meta["quantization_annotation"].input_qspec_map[last_node]
    if len(list(last_node.users.keys())) == 1 or _all_users_annotate_equal(last_node, input_qspec):
        if "quantization_annotation" in last_node.meta:
            if isinstance(last_node.meta["quantization_annotation"].output_qspec, SharedQuantizationSpec):
                prev_node = last_node.meta["quantization_annotation"].output_qspec.edge_or_node
                while isinstance(prev_node.meta["quantization_annotation"].output_qspec, SharedQuantizationSpec)\
                    and (len(list(prev_node.users.keys())) == 1 or _all_users_annotate_equal(prev_node, input_qspec)):
                    prev_node = prev_node.meta["quantization_annotation"].output_qspec.edge_or_node
                if len(list(prev_node.users.keys())) == 1 or _all_users_annotate_equal(prev_node, input_qspec):
                    prev_node.meta["quantization_annotation"].output_qspec = output_qspec
            elif isinstance(last_node.meta["quantization_annotation"].output_qspec, QuantizationSpec):
                last_node.meta["quantization_annotation"].output_qspec = output_qspec
            else:
                assert False
    return


def _mark_nodes_as_annotated(nodes: List[Node]):
    for node in nodes:
        if node is not None:
            if "quantization_annotation" not in node.meta:
                node.meta["quantization_annotation"] = QuantizationAnnotation()
            node.meta["quantization_annotation"]._annotated = True


def _is_input_large_scalar(node: Node, gm: torch.fx.GraphModule):
    """Check if input is a large scalar value. So that we can skip quantization for the node
    since histc op (in HistogramObserver) only works for values up to certain upper bound
    """
    if node.op == "get_attr":
        qualified_name = str(node.target)
        module_path, _, name = qualified_name.rpartition(".")
        submod = gm.get_submodule(module_path)
        tensor = getattr(submod, name)
        # torch.histc works until this upper bound
        HISTC_UPPER_BOUND = 3.4028235e15
        return tensor.numel() == 1 and abs(tensor.item()) > HISTC_UPPER_BOUND
    return False


def _is_input_non_float_tensor(node: Node):
    """Check if the input is not a float tensor, so that we can skip quantization for the node
    since observers only works with float Tensors
    """
    if "val" not in node.meta or not isinstance(node.meta["val"], FakeTensor):
        return True
    return node.meta["val"].dtype != torch.float32


def get_input_act_qspec(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    if quantization_config.input_activation is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.input_activation
    assert quantization_spec.qscheme in [
        torch.per_tensor_affine,
        torch.per_tensor_symmetric,
    ]
    return quantization_spec


def get_output_act_qspec(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    if quantization_config.output_activation is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.output_activation
    assert quantization_spec.qscheme in [
        torch.per_tensor_affine,
        torch.per_tensor_symmetric,
    ]
    return quantization_spec

from torch.ao.quantization.observer import _PartialWrapper
import functools
def get_weight_shape(gm:torch.fx.GraphModule, weight_node):
    weight_tensor_shape = None
    if 'val' in weight_node.meta and weight_node.meta['val'] is not None:
        weight_tensor_shape = weight_node.meta['val'].shape
    elif 'fake_tensor' in weight_node.meta and weight_node.meta['fake_tensor'] is not None:
        weight_tensor_shape = weight_node.meta['fake_tensor'].shape
    elif weight_node.op == 'get_attr':
        # If weight_node is a 'get_attr' operation, its target is the name of the attribute
        # on the GraphModule (e.g., 'linear1.weight' or '_param_constant0').
        # We can retrieve the actual tensor from the GraphModule.
        try:
            weight_tensor = getattr(gm, weight_node.target)
            weight_tensor_shape = weight_tensor.shape
        except AttributeError:
            print(f"  Warning: Could not find attribute '{weight_node.target}' on GraphModule for weight_node '{weight_node.name}'.")
    elif weight_node.op == 'placeholder':
        # If weight_node is a 'placeholder', its shape should be in its meta['tensor_meta']
        if 'tensor_meta' in weight_node.meta and weight_node.meta['tensor_meta'] is not None:
            weight_tensor_shape = weight_node.meta['tensor_meta'].shape

    return weight_tensor_shape
def _ctr_is_fakequat(obj, fake_quant_class:FakeQuantizeBase=_LearnableFakeQuantize):
    """
    检查一个对象是否是 _LearnableFakeQuantize 类、其实例，
    或者一个包装了 _LearnableFakeQuantize 类的 functools.partial 对象。
    """
    if isinstance(obj, _PartialWrapper):
        wrapped_callable = obj.p # Access the wrapped callable
        if isinstance(wrapped_callable, functools.partial):
            return wrapped_callable.func == fake_quant_class
        elif wrapped_callable == fake_quant_class: # If it directly wraps the class
            return True
    elif isinstance(obj, functools.partial):
        # 如果是 partial 对象，检查其 func 属性
        return obj.func == fake_quant_class
    elif isinstance(obj, type):
        # 如果是类本身
        return obj == fake_quant_class
    elif isinstance(obj, fake_quant_class):
        # 如果是 _LearnableFakeQuantize 的实例
        return True
    return False

def get_weight_qspec(quantization_config: Optional[QuantizationConfig], weight_node_shape:List[int]=None):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    if quantization_config.weight is None:
        return 
    if _ctr_is_fakequat(quantization_config.weight.observer_or_fake_quant_ctr, _LearnableFakeQuantize) :
        quantization_spec: QuantizationSpec = quantization_config.weight
        if quantization_spec.qscheme not in [
            torch.per_tensor_symmetric,
            torch.per_channel_symmetric,
            None,
        ]:
            raise ValueError(
                f"Unsupported quantization_spec {quantization_spec} for weight"
            )
        assert weight_node_shape is not None, f"weight shape can't be None"
        ch_axis = quantization_spec.ch_axis
        channel_len = weight_node_shape[ch_axis]
        extra_args = quantization_config.weight.observer_or_fake_quant_ctr.p.keywords
        # extra_args["use_grad_scaling"] = True
        extra_args["channel_len"] = channel_len
        quantization_spec = QuantizationSpec(
            dtype=quantization_config.weight.dtype,
            quant_min=quantization_config.weight.quant_min,
            quant_max=quantization_config.weight.quant_max,
            qscheme=quantization_config.weight.qscheme,
            ch_axis=ch_axis,
            is_dynamic=False,
            observer_or_fake_quant_ctr=_LearnableFakeQuantize.with_args(
                **extra_args
            ),
        )
        return quantization_spec
    quantization_spec: QuantizationSpec = quantization_config.weight
    if quantization_spec.qscheme not in [
        torch.per_tensor_symmetric,
        torch.per_channel_symmetric,
        None,
    ]:
        raise ValueError(
            f"Unsupported quantization_spec {quantization_spec} for weight"
        )
    return quantization_spec    
# def get_weight_qspec(quantization_config: Optional[QuantizationConfig]):
#     if quantization_config is None:
#         return None
#     assert quantization_config is not None
#     if quantization_config.weight is None:
#         return None
#     quantization_spec: QuantizationSpec = quantization_config.weight
#     if quantization_spec.qscheme not in [
#         torch.per_tensor_symmetric,
#         torch.per_channel_symmetric,
#         None,
#     ]:
#         raise ValueError(
#             f"Unsupported quantization_spec {quantization_spec} for weight"
#         )
#     return quantization_spec


def get_weight_trans_qspec(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    if quantization_config.weight is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.weight_trans
    if quantization_spec.qscheme not in [
        torch.per_tensor_symmetric,
        torch.per_channel_symmetric,
        None,
    ]:
        raise ValueError(
            f"Unsupported quantization_spec {quantization_spec} for weight"
        )
    return quantization_spec


def get_bias_qspec(quantization_config: Optional[QuantizationConfig]):
    if quantization_config is None:
        return None
    assert quantization_config is not None
    if quantization_config.bias is None:
        return None
    quantization_spec: QuantizationSpec = quantization_config.bias
    assert (
        quantization_spec.dtype == torch.float
    ), "Only float dtype for bias is supported for bias right now"
    return quantization_spec


@register_annotator("linear")
def _annotate_linear(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:

    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
            continue

        linear_node = node
        weight_node = linear_node.args[1]
        bias_node = None
        if len(linear_node.args) > 2:
            bias_node = linear_node.args[2]

        partition = [linear_node, weight_node]
        if bias_node is not None:
            partition.append(bias_node)
        output_node = linear_node

        if len(list(linear_node.users.keys())) == 1 and list(linear_node.users.keys())[0].target in [
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
        ]:
            relu_node = list(linear_node.users.keys())[0]
            output_node = relu_node
            partition.append(relu_node)

        input_node = linear_node.args[0]
        if is_global:
            if _is_annotated(partition):
                continue
            input_qspec_map = {}
            input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
            input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
            if bias_node is not None:
                input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
            linear_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
            )
            if output_node == linear_node:
                linear_node.meta["quantization_annotation"].output_qspec = get_output_act_qspec(quantization_config)
            else:
                output_node.meta["quantization_annotation"] = QuantizationAnnotation(
                    output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                    _annotated=True,
                )
            _mark_nodes_as_annotated(partition)
        else:
            if module_names is not None and linear_node.name not in module_names:
                continue
            if not _is_annotated(partition):
                assert False
            # Annotate node inputs and last node output
            input_qspec_map = {}
            input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
            input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
            if bias_node is not None:
                input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
            linear_node.meta["quantization_annotation"].input_qspec_map = input_qspec_map
            _update_last_node_output_qspec(input_node, linear_node, get_input_act_qspec(quantization_config))
    return


@register_annotator("conv")
def _annotate_conv(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
):
    """
    Given a function that takes in a `conv_fn` and returns a conv-bn[-relu] pattern,
    return a list of annotated partitions.

    The output of the pattern must include a dictionary from string name to node
    for the following names: "input", "conv", "weight", "bias", and "output".
    """

    def get_pattern(conv_fn: Callable, has_bn: bool, has_relu: bool, relu_is_inplace: bool):
        def _conv_bn(x, conv_weight, conv_bias, bn_weight=None, bn_bias=None, bn_rm=None, bn_rv=None):
            conv = conv_fn(x, conv_weight, conv_bias)
            if has_bn:
                bn = F.batch_norm(conv, bn_rm, bn_rv, bn_weight, bn_bias, training=True)
            else:
                bn = conv
            if has_relu:
                output = F.relu_(bn) if relu_is_inplace else F.relu(bn)
            else:
                output = bn
            return output, {
                "input": x,
                "conv": conv,
                "weight": conv_weight,
                "bias": conv_bias,
                "output": output,
            }

        return _WrapperModule(_conv_bn)

    # Needed for matching, otherwise the matches gets filtered out due to unused
    # nodes returned by batch norm
    gm.graph.eliminate_dead_code()
    gm.recompile()

    from torch._export import gm_using_training_ir

    using_training_ir = gm_using_training_ir(gm)

    # example_inputs
    _conv1d_example_inputs = (
        torch.randn(1, 1, 3),  # x
        torch.randn(1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
    )
    _conv1d_bn_example_inputs = (
        torch.randn(1, 1, 3),  # x
        torch.randn(1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )
    _conv2d_example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1, 1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
    )
    _conv2d_bn_example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1, 1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )

    matches: List[InternalMatch] = []
    combinations = [
        (F.conv1d, False, _conv1d_example_inputs),  # conv_fn, has_bn, example_input
        (F.conv1d, True, _conv1d_bn_example_inputs),  # conv_fn, has_bn, example_input
        (F.conv2d, False, _conv2d_example_inputs),  # type: ignore[list-item]
        (F.conv2d, True, _conv2d_bn_example_inputs),  # type: ignore[list-item]
    ]

    # Add `is_cuda` and `relu_is_inplace` dimensions
    combinations = itertools.product(  # type: ignore[assignment]
        combinations,
        [True] if torch.cuda.is_available() else [False],  # is_cuda
        [True, False],  # has_relu
        [True, False],  # relu_is_inplace
    )

    # Match against all conv dimensions and cuda variants
    for (conv_fn, has_bn, example_inputs), is_cuda, has_relu, relu_is_inplace in combinations:  # type: ignore[misc]
        if not has_relu and relu_is_inplace:
            continue
        pattern = get_pattern(conv_fn, has_bn, has_relu, relu_is_inplace)  # type: ignore[has-type]
        pattern = _get_aten_graph_module_for_pattern(pattern, example_inputs, is_cuda, using_training_ir=using_training_ir)  # type: ignore[has-type]
        pattern.graph.eliminate_dead_code()
        pattern.recompile()
        matcher = SubgraphMatcherWithNameNodeMap(pattern, ignore_literals=True)
        sub_matches = matcher.match(gm.graph)
        for sub_match in sub_matches:
            if not has_relu:
                output_node = sub_match.name_node_map["output"]
                users = list(output_node.users.keys())
                if len(users) == 1:
                    next_node = users[0]
                    if next_node.op == "call_function" and next_node.target in [
                        torch.ops.aten.relu.default,
                        torch.ops.aten.relu_.default,
                    ]:
                        continue
                    if not has_bn:
                        if next_node.op == "call_function" and next_node.target in [
                            torch.ops.aten.batch_norm.default
                        ]:
                            continue
            else:
                if not has_bn:
                    # hack: relu 的上一个算子被 match 到的也是 relu ？
                    sub_match.name_node_map["conv"] = sub_match.name_node_map["output"].args[0]
            matches.append(sub_match)

    # Annotate nodes returned in the matches
    for match in matches:
        name_node_map = match.name_node_map
        input_node = name_node_map["input"]
        conv_node = name_node_map["conv"]
        weight_node = name_node_map["weight"]
        bias_node = name_node_map["bias"]
        output_node = name_node_map["output"]

        # TODO: annotate the uses of input, weight, and bias separately instead
        # of assuming they come from a single conv node. This is not possible today
        # because input may have multiple users, and we can't rely on the conv node
        # always being the first user. This was the case in models with skip
        # connections like resnet18

        # Validate conv args
        if conv_node.args[0] is not input_node:
            raise ValueError("Conv arg did not contain input node ", input_node)
        if conv_node.args[1] is not weight_node:
            raise ValueError("Conv arg did not contain weight node ", weight_node)
        if len(conv_node.args) > 2 and conv_node.args[2] is not bias_node:
            raise ValueError("Conv arg did not contain bias node ", bias_node)

        # Skip if the partition is already annotated or is filtered out by the user
        partition = [conv_node, weight_node]
        if bias_node is not None:
            partition.append(bias_node)

        if is_global:
            # if _is_annotated(partition):
            #     continue
            if not _is_annotated(partition[:1]) and _is_annotated(partition[1:]):
                annotated_conv = None
                for node in weight_node.users:
                    if _is_annotated([node]):
                        annotated_conv = node
                        break
                annotated_output = annotated_conv
                while "quantization_annotation" not in annotated_output.meta \
                    or annotated_output.meta["quantization_annotation"].output_qspec is None:
                    annotated_output = list(annotated_output.users.keys())[0]

                input_qspec_map = {}
                input_qspec_map[input_node] = SharedQuantizationSpec((annotated_conv.args[0], annotated_conv))
                input_qspec_map[weight_node] = SharedQuantizationSpec((weight_node, annotated_conv))
                if bias_node is not None:
                    input_qspec_map[bias_node] = SharedQuantizationSpec((bias_node, annotated_conv))
                conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map=input_qspec_map,
                    _annotated=True,
                )
                if output_node == conv_node:
                    conv_node.meta["quantization_annotation"].output_qspec = SharedQuantizationSpec((annotated_conv, list(annotated_conv.users.keys())[0]))
                else:
                    output_node.meta["quantization_annotation"] = QuantizationAnnotation(
                        output_qspec=SharedQuantizationSpec((annotated_output, list(annotated_output.users.keys())[0])),  # type: ignore[arg-type]
                        _annotated=True,
                    )
                _mark_nodes_as_annotated([conv_node])
            elif not _is_annotated(partition):
                # Annotate conv inputs and pattern output
                input_qspec_map = {}
                input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
                weight_node_shape = get_weight_shape(gm, weight_node)
                input_qspec_map[weight_node] = get_weight_qspec(quantization_config, weight_node_shape)
                if bias_node is not None:
                    input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
                conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map=input_qspec_map,
                    _annotated=True,
                )
                if output_node == conv_node:
                    conv_node.meta["quantization_annotation"].output_qspec = get_output_act_qspec(quantization_config)
                else:
                    output_node.meta["quantization_annotation"] = QuantizationAnnotation(
                        output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                        _annotated=True,
                    )
                _mark_nodes_as_annotated(partition)
            else:
                continue
        else:
            if module_names is not None and conv_node.name not in module_names:
                continue
            if not _is_annotated(partition):
                assert False
            # Annotate node inputs and last node output
            old_input_qspec_map = conv_node.meta["quantization_annotation"].input_qspec_map
            input_qspec_map = {}
            input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
            input_qspec_map[weight_node] = old_input_qspec_map[weight_node] \
                if isinstance(old_input_qspec_map[weight_node], SharedQuantizationSpec) else get_weight_qspec(quantization_config)
            if bias_node is not None:
                input_qspec_map[bias_node] = old_input_qspec_map[bias_node] \
                    if isinstance(old_input_qspec_map[bias_node], SharedQuantizationSpec) else get_bias_qspec(quantization_config)
            conv_node.meta["quantization_annotation"].input_qspec_map = input_qspec_map
            _update_last_node_output_qspec(input_node, conv_node, get_input_act_qspec(quantization_config))
    return


@register_annotator("convtranspose")
def _annotate_convtranspose(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
):
    def get_pattern(conv_fn: Callable, has_bn: bool, has_relu: bool, relu_is_inplace: bool):
        def _conv_bn(x, conv_weight, conv_bias, bn_weight=None, bn_bias=None, bn_rm=None, bn_rv=None):
            conv = conv_fn(x, conv_weight, conv_bias)
            if has_bn:
                bn = F.batch_norm(conv, bn_rm, bn_rv, bn_weight, bn_bias, training=True)
            else:
                bn = conv
            if has_relu:
                output = F.relu_(bn) if relu_is_inplace else F.relu(bn)
            else:
                output = bn
            return output, {
                "input": x,
                "conv": conv,
                "weight": conv_weight,
                "bias": conv_bias,
                "output": output,
            }

        return _WrapperModule(_conv_bn)

    # Needed for matching, otherwise the matches gets filtered out due to unused
    # nodes returned by batch norm
    gm.graph.eliminate_dead_code()
    gm.recompile()

    from torch._export import gm_using_training_ir

    using_training_ir = gm_using_training_ir(gm)

    # example_inputs
    _conv1d_example_inputs = (
        torch.randn(1, 1, 3),  # x
        torch.randn(1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
    )
    _conv1d_bn_example_inputs = (
        torch.randn(1, 1, 3),  # x
        torch.randn(1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )
    _conv2d_example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1, 1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
    )
    _conv2d_bn_example_inputs = (
        torch.randn(1, 1, 3, 3),  # x
        torch.randn(1, 1, 1, 1),  # conv_weight
        torch.randn(1),  # conv_bias
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_running_mean
        torch.randn(1),  # bn_running_var
    )

    matches: List[InternalMatch] = []
    combinations = [
        (F.conv_transpose1d, False, _conv1d_example_inputs),  # conv_fn, has_bn, example_input
        (F.conv_transpose1d, True, _conv1d_bn_example_inputs),  # conv_fn, has_bn, example_input
        (F.conv_transpose2d, False, _conv2d_example_inputs),  # type: ignore[list-item]
        (F.conv_transpose2d, True, _conv2d_bn_example_inputs),  # type: ignore[list-item]
    ]

    # Add `is_cuda` and `relu_is_inplace` dimensions
    combinations = itertools.product(  # type: ignore[assignment]
        combinations,
        [True, False] if torch.cuda.is_available() else [False],  # is_cuda
        [True, False],  # has_relu
        [True, False],  # relu_is_inplace
    )

    # Match against all conv dimensions and cuda variants
    for (conv_fn, has_bn, example_inputs), is_cuda, has_relu, relu_is_inplace in combinations:  # type: ignore[misc]
        if not has_relu and relu_is_inplace:
            continue
        pattern = get_pattern(conv_fn, has_bn, has_relu, relu_is_inplace)  # type: ignore[has-type]
        pattern = _get_aten_graph_module_for_pattern(pattern, example_inputs, is_cuda, using_training_ir=using_training_ir)  # type: ignore[has-type]
        pattern.graph.eliminate_dead_code()
        pattern.recompile()
        matcher = SubgraphMatcherWithNameNodeMap(pattern, ignore_literals=True)
        sub_matches = matcher.match(gm.graph)
        for sub_match in sub_matches:
            if not has_relu:
                output_node = sub_match.name_node_map["output"]
                users = list(output_node.users.keys())
                if len(users) == 1:
                    next_node = users[0]
                    if next_node.op == "call_function" and next_node.target in [
                        torch.ops.aten.relu.default,
                        torch.ops.aten.relu_.default,
                    ]:
                        continue
                    if not has_bn:
                        if next_node.op == "call_function" and next_node.target in [
                            torch.ops.aten.batch_norm.default
                        ]:
                            continue
            else:
                if not has_bn:
                    # hack: relu 的上一个算子被 match 到的也是 relu ？
                    sub_match.name_node_map["conv"] = sub_match.name_node_map["output"].args[0]
            matches.append(sub_match)

    # Annotate nodes returned in the matches
    for match in matches:
        name_node_map = match.name_node_map
        input_node = name_node_map["input"]
        conv_node = name_node_map["conv"]
        weight_node = name_node_map["weight"]
        bias_node = name_node_map["bias"]
        output_node = name_node_map["output"]

        # TODO: annotate the uses of input, weight, and bias separately instead
        # of assuming they come from a single conv node. This is not possible today
        # because input may have multiple users, and we can't rely on the conv node
        # always being the first user. This was the case in models with skip
        # connections like resnet18

        # Validate conv args
        if conv_node.args[0] is not input_node:
            raise ValueError("Conv arg did not contain input node ", input_node)
        if conv_node.args[1] is not weight_node:
            raise ValueError("Conv arg did not contain weight node ", weight_node)
        if len(conv_node.args) > 2 and conv_node.args[2] is not bias_node:
            raise ValueError("Conv arg did not contain bias node ", bias_node)

        # Skip if the partition is already annotated or is filtered out by the user
        partition = [conv_node, weight_node]
        if bias_node is not None:
            partition.append(bias_node)

        if is_global:
            if _is_annotated(partition):
                continue
            # Annotate conv inputs and pattern output
            input_qspec_map = {}
            input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
            input_qspec_map[weight_node] = get_weight_trans_qspec(quantization_config)
            if bias_node is not None:
                input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
            conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
            )
            if output_node == conv_node:
                conv_node.meta["quantization_annotation"].output_qspec = get_output_act_qspec(quantization_config)
            else:
                output_node.meta["quantization_annotation"] = QuantizationAnnotation(
                    output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
                    _annotated=True,
                )
            _mark_nodes_as_annotated(partition)
        else:
            if module_names is not None and conv_node.name not in module_names:
                continue
            if not _is_annotated(partition):
                assert False
            # Annotate node inputs and last node output
            input_qspec_map = {}
            input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
            input_qspec_map[weight_node] = get_weight_trans_qspec(quantization_config)
            if bias_node is not None:
                input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
            conv_node.meta["quantization_annotation"].input_qspec_map = input_qspec_map
            _update_last_node_output_qspec(input_node, conv_node, get_input_act_qspec(quantization_config))
    return


@register_annotator("gru_io_only")
def _annotate_gru_io_only(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    gru_partitions = get_source_partitions(gm.graph, [torch.nn.GRU], filter_fn)
    gru_partitions = list(itertools.chain.from_iterable(gru_partitions.values()))
    annotated_partitions = []
    for gru_partition in gru_partitions:
        annotated_partitions.append(gru_partition.nodes)
        output_nodes = gru_partition.output_nodes
        input_nodes = gru_partition.input_nodes
        # skip annotation if it is already annotated
        if _is_annotated(input_nodes + output_nodes):
            continue
        # inside each GRU partition, we should be able to annotate each linear
        # subgraph
        input_qspec_map: Dict[Node, QuantizationSpecBase] = {}
        input_act = input_nodes[0]
        input_act_user = next(iter(input_act.users.keys()))
        assert isinstance(input_act, Node)
        assert isinstance(input_act_user, Node)
        input_act_user.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                input_act: get_input_act_qspec(quantization_config),
            },
            _annotated=True,
        )

        hidden_state = input_nodes[1]
        hidden_state_user = next(iter(hidden_state.users.keys()))
        assert isinstance(hidden_state, Node)
        assert isinstance(hidden_state_user, Node)
        hidden_state_user.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                hidden_state: get_input_act_qspec(quantization_config),
            },
            _annotated=True,
        )

        assert len(output_nodes) == 2, "expecting GRU to have two outputs"
        for output in output_nodes:
            output.meta["quantization_annotation"] = QuantizationAnnotation(
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )
        nodes_to_mark_annotated = list(gru_partition.nodes)
        _mark_nodes_as_annotated(nodes_to_mark_annotated)
    return annotated_partitions


@register_annotator("avgpool2d")
def _annotate_adaptive_avg_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:
    """Always annotate adaptive_avg_pool2d op"""
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d], None
    )
    partitions = list(itertools.chain.from_iterable(module_partitions.values()))

    for partition in partitions:
        pool_node = partition.output_nodes[0]
        if (
            pool_node.op != "call_function"
            or pool_node.target != torch.ops.aten.adaptive_avg_pool2d.default
        ):
            raise ValueError(f"{pool_node} is not an aten adaptive_avg_pool2d operator")

        input_node = pool_node.args[0]
        assert isinstance(input_node, Node)

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_qspec_map[input_node] = input_act_qspec

        if is_global:
            if _is_annotated([pool_node]):
                continue
            pool_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_act_qspec,
                _annotated=True,
            )
        else:
            if not _is_annotated(partition):
                assert False
            if module_names is not None and pool_node.name not in module_names:
                continue

            pool_node.meta["quantization_annotation"].input_qspec_map = input_qspec_map
            _update_last_node_output_qspec(input_node, pool_node, get_input_act_qspec(quantization_config))
    return


@register_annotator("layernorm")
def _annotate_layer_norm(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:
    """Always annotate layer_norm op"""
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.LayerNorm, F.layer_norm], None
    )
    partitions = list(itertools.chain.from_iterable(module_partitions.values()))

    for partition in partitions:
        norm_node = partition.output_nodes[0]
        if (
            norm_node.op != "call_function"
            or norm_node.target != torch.ops.aten.layer_norm.default
        ):
            raise ValueError(f"{norm_node} is not an aten adaptive_avg_pool2d operator")

        input_node = norm_node.args[0]
        assert isinstance(input_node, Node)

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_qspec_map[input_node] = input_act_qspec

        if is_global:
            if _is_annotated([norm_node]):
                continue
            norm_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_act_qspec,
                _annotated=True,
            )
        else:
            if not _is_annotated(partition):
                assert False
            if module_names is not None and norm_node.name not in module_names:
                continue

            norm_node.meta["quantization_annotation"].input_qspec_map = input_qspec_map
            _update_last_node_output_qspec(input_node, norm_node, get_input_act_qspec(quantization_config))
    return


@register_annotator("groupnorm")
def _annotate_group_norm(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:
    """Always annotate group_norm op"""
    module_partitions = get_source_partitions(
        gm.graph, [torch.nn.GroupNorm, F.group_norm], None
    )
    partitions = list(itertools.chain.from_iterable(module_partitions.values()))

    for partition in partitions:
        norm_node = partition.output_nodes[0]
        if (
            norm_node.op != "call_function"
            or norm_node.target != torch.ops.aten.group_norm.default
        ):
            raise ValueError(f"{norm_node} is not an aten adaptive_avg_pool2d operator")

        input_node = norm_node.args[0]
        assert isinstance(input_node, Node)

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_qspec_map[input_node] = input_act_qspec

        if is_global:
            if _is_annotated([norm_node]):
                continue
            norm_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_act_qspec,
                _annotated=True,
            )
        else:
            if not _is_annotated(partition):
                assert False
            if module_names is not None and norm_node.name not in module_names:
                continue

            norm_node.meta["quantization_annotation"].input_qspec_map = input_qspec_map
            _update_last_node_output_qspec(input_node, norm_node, get_input_act_qspec(quantization_config))
    return


def _do_annotate_dyadic(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
    aten_ops: List[torch._ops.OpOverload] = [],
) -> Optional[List[List[Node]]]:

    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in aten_ops:
            continue
        dyadic_node = node
        output_node = dyadic_node
        partition = [dyadic_node]

        input_node0 = dyadic_node.args[0]
        input_node1 = dyadic_node.args[1]

        if len(dyadic_node.users) == 1 and list(dyadic_node.users.keys())[0].target in [
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
        ]:
            relu_node = list(dyadic_node.users.keys())[0]
            output_node = relu_node
            partition.append(relu_node)

        
        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)
        input_qspec_map = {}
        if isinstance(input_node0, Node):
            if _is_input_large_scalar(input_node0, gm):
                continue
            if _is_input_non_float_tensor(input_node0):
                continue
            input_qspec_map[input_node0] = input_act_qspec
        if isinstance(input_node1, Node):
            if _is_input_large_scalar(input_node1, gm):
                continue
            if _is_input_non_float_tensor(input_node1):
                continue
            input_qspec_map[input_node1] = input_act_qspec

        if is_global:
            if _is_annotated(partition):
                continue
            dyadic_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                _annotated=True,
            )
            if output_node == dyadic_node:
                dyadic_node.meta["quantization_annotation"].output_qspec=output_act_qspec
            else:
                output_node.meta["quantization_annotation"] = QuantizationAnnotation(
                    output_qspec=output_act_qspec,
                    _annotated=True,
                )
        else:
            if not _is_annotated(partition):
                assert False
            if module_names is not None and dyadic_node.name not in module_names:
                continue

            dyadic_node.meta["quantization_annotation"].input_qspec_map = input_qspec_map
            _update_last_node_output_qspec(input_node0, dyadic_node, get_input_act_qspec(quantization_config))
            _update_last_node_output_qspec(input_node1, dyadic_node, get_input_act_qspec(quantization_config))
    return


@register_annotator("add")
def _annotate_add(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:

    aten_ops = [
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add_.Tensor,
    ]
    _do_annotate_dyadic(
        gm,
        quantization_config,
        module_names,
        is_global,
        aten_ops
    )
    

@register_annotator("mul")
def _annotate_mul(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:

    aten_ops = [
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul_.Tensor,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.mul_.Scalar,
    ]
    _do_annotate_dyadic(
        gm,
        quantization_config,
        module_names,
        is_global,
        aten_ops
    )


@register_annotator("matmul")
def _annotate_matmul(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:

    aten_ops = [
        torch.ops.aten.matmul.default,
    ]
    _do_annotate_dyadic(
        gm,
        quantization_config,
        module_names,
        is_global,
        aten_ops
    )


@register_annotator("gridsample")
def _annotate_gridsample(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:

    aten_ops = [
        torch.ops.aten.grid_sampler.default,
    ]
    _do_annotate_dyadic(
        gm,
        quantization_config,
        module_names,
        is_global,
        aten_ops
    )


def _do_annotate_activate(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
    aten_ops: List[torch._ops.OpOverload] = [],
) -> Optional[List[List[Node]]]:

    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in aten_ops:
            continue

        act_node = node
        partition = [act_node]

        input_node = act_node.args[0]
        if _is_input_large_scalar(input_node, gm):
            continue
        if _is_input_non_float_tensor(input_node):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_qspec_map[input_node] = input_act_qspec

        if is_global:
            if _is_annotated(partition):
                continue
            act_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_act_qspec,
                _annotated=True,
            )
        else:
            if not _is_annotated(partition):
                assert False
            if module_names is not None and act_node.name not in module_names:
                continue

            act_node.meta["quantization_annotation"].input_qspec_map = input_qspec_map
            _update_last_node_output_qspec(input_node, act_node, get_input_act_qspec(quantization_config))
    return


@register_annotator("silu")
def _annotate_silu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:

    aten_ops = [
        torch.ops.aten.silu.default,
        torch.ops.aten.silu_.default,
    ]
    _do_annotate_activate(
        gm,
        quantization_config,
        module_names,
        is_global,
        aten_ops
    )


@register_annotator("gelu")
def _annotate_gelu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:

    aten_ops = [
        torch.ops.aten.gelu.default,
        torch.ops.aten.gelu_.default,
    ]
    _do_annotate_activate(
        gm,
        quantization_config,
        module_names,
        is_global,
        aten_ops
    )


@register_annotator("glu")
def _annotate_glu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:

    aten_ops = [
        torch.ops.aten.glu.default,
    ]
    _do_annotate_activate(
        gm,
        quantization_config,
        module_names,
        is_global,
        aten_ops
    )


@register_annotator("softmax")
def _annotate_softmax(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:

    aten_ops = [
        torch.ops.aten._safe_softmax.default,
        torch.ops.aten.softmax.int,
    ]
    _do_annotate_activate(
        gm,
        quantization_config,
        module_names,
        is_global,
        aten_ops
    )


# TODO: remove Optional in return type, fix annotated_partitions logic
@register_annotator("concat")
def _annotate_cat(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:
    cat_partitions = get_source_partitions(gm.graph, [torch.cat], None)
    cat_partitions = list(itertools.chain.from_iterable(cat_partitions.values()))

    for cat_partition in cat_partitions:
        cat_node = cat_partition.output_nodes[0]

        if cat_node.target != torch.ops.aten.cat.default:
            # TODO: change this to AnnotationException
            raise Exception(  # noqa: TRY002
                f"Expected cat node: torch.ops.aten.cat.default, but found {cat_node.target}"
                " please check if you are calling the correct capture API"
            )

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        inputs = cat_node.args[0]
        input_qspec_map = {}
        for input_act in inputs:
            input_qspec_map[input_act] = input_act_qspec

        if is_global:
            if _is_annotated([cat_node]):
                continue
            cat_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_act_qspec,
                _annotated=True,
            )
        else:
            if not _is_annotated([cat_node]):
                assert False
            if module_names is not None and cat_node.name not in module_names:
                continue

            cat_node.meta["quantization_annotation"].input_qspec_map = input_qspec_map
            for input_node in inputs:
                _update_last_node_output_qspec(input_node, cat_node, get_input_act_qspec(quantization_config))
    return

    
@register_annotator("sdpa")
def _annotate_scaled_dot_product_attention(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:

    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in [torch.ops.aten.scaled_dot_product_attention.default]:
            continue

        act_node = node
        partition = [act_node]

        input_node_q = act_node.args[0]
        input_node_k = act_node.args[1]
        input_node_v = act_node.args[2]

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {
            input_node_q: input_act_qspec,
            input_node_k: input_act_qspec,
            input_node_v: input_act_qspec,
        }

        if is_global:
            if _is_annotated(partition):
                continue
            act_node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_act_qspec,
                _annotated=True,
            )
        else:
            if not _is_annotated(partition):
                assert False
            if module_names is not None and act_node.name not in module_names:
                continue

            act_node.meta["quantization_annotation"].input_qspec_map = input_qspec_map
            _update_last_node_output_qspec(input_node_q, act_node, get_input_act_qspec(quantization_config))
            _update_last_node_output_qspec(input_node_k, act_node, get_input_act_qspec(quantization_config))
            _update_last_node_output_qspec(input_node_v, act_node, get_input_act_qspec(quantization_config))
    return


@register_annotator("mha")
def _annotate_mha(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:
    mha_partitions = get_source_partitions(gm.graph, [torch.nn.modules.activation.MultiheadAttention], None)
    mha_partitions = list(itertools.chain.from_iterable(mha_partitions.values()))

    # from IPython import embed; embed()
    for mha_partition in mha_partitions:
        _mark_nodes_as_annotated(mha_partition.nodes)
        # cat_node = mha_partition.output_nodes[0]

        # if cat_node.target != torch.ops.aten.cat.default:
        #     # TODO: change this to AnnotationException
        #     raise Exception(  # noqa: TRY002
        #         f"Expected cat node: torch.ops.aten.cat.default, but found {cat_node.target}"
        #         " please check if you are calling the correct capture API"
        #     )

        # input_act_qspec = get_input_act_qspec(quantization_config)
        # output_act_qspec = get_output_act_qspec(quantization_config)

        # inputs = cat_node.args[0]
        # input_qspec_map = {}
        # for input_act in inputs:
        #     input_qspec_map[input_act] = input_act_qspec

        # if is_global:
        #     if _is_annotated([cat_node]):
        #         continue
        #     cat_node.meta["quantization_annotation"] = QuantizationAnnotation(
        #         input_qspec_map=input_qspec_map,
        #         output_qspec=output_act_qspec,
        #         _annotated=True,
        #     )
        # else:
        #     if not _is_annotated([cat_node]):
        #         assert False
        #     if module_names is not None and cat_node.name not in module_names:
        #         continue

        #     cat_node.meta["quantization_annotation"].input_qspec_map = input_qspec_map
        #     for input_node in inputs:
        #         _update_last_node_output_qspec(input_node, cat_node, get_input_act_qspec(quantization_config))
    return


@register_annotator("split")
def _annotate_split(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    module_names: List[str] = None,
    is_global: bool = True,
) -> Optional[List[List[Node]]]:
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.split_with_sizes.default,
            torch.ops.aten.chunk.default,
        ]:
            continue

        split_node = node
        partition = [split_node]

        users = []
        for user in split_node.users.keys():
            if (
                isinstance(user, Node)
                and user.op == "call_function"
                and user.target in [
                    operator.getitem,
                ]
            ):
                users.append(user)
                partition.append(user)

        if is_global:
            if _is_annotated(partition):
                continue

            prev_node = split_node.args[0]
            if prev_node.op == "placeholder":
                shared_qspec = SharedQuantizationSpec((prev_node, split_node))
                split_node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={
                    split_node: get_input_act_qspec(quantization_config),
                },
                    _annotated=True,
                )
                for user in users:
                    user.meta["quantization_annotation"] = QuantizationAnnotation(
                        output_qspec=shared_qspec,
                        _annotated=True,
                    )
            else:
                shared_qspec = SharedQuantizationSpec(prev_node)
                split_node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={
                    split_node: shared_qspec,
                },
                    _annotated=True,
                )
                for user in users:
                    user.meta["quantization_annotation"] = QuantizationAnnotation(
                        output_qspec=shared_qspec,
                        _annotated=True,
                    )
        else:
            assert False
    return


def _is_share_obs_or_fq_op(op: Callable) -> bool:
    return op in [
        # maxpool
        torch.ops.aten.max_pool2d.default,
        # identity
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        # transpose
        torch.ops.aten.permute.default,
        torch.ops.aten.permute_copy.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.t.default,
        # reshape
        torch.ops.aten.unflatten.int,
        torch.ops.aten.view_copy.default,
        torch.ops.aten.view.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.reshape.default,
        # squeeze
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze_copy.dim,
        torch.ops.aten.unsqueeze.default,
        # resize
        torch.ops.aten.upsample_nearest1d.vec,
        torch.ops.aten.upsample_nearest2d.vec,
        torch.ops.aten.upsample_nearest3d.vec,
        torch.ops.aten.upsample_linear1d.vec,
        torch.ops.aten.upsample_bilinear2d.vec,
        torch.ops.aten.upsample_bicubic2d.vec,
        torch.ops.aten.upsample_trilinear3d.vec,
        # gather
        torch.ops.aten.select.int,
        # pixel shuffle/unshuffle
        torch.ops.aten.pixel_shuffle.default,
        torch.ops.aten.pixel_unshuffle.default,
        # others
        torch.ops.aten.relu.default,
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
        torch.ops.aten.mean.default,
        torch.ops.aten.mean.dim,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.slice_copy.Tensor,
    ]


def propagate_annotation(model: torch.fx.GraphModule, quantization_config: Optional[QuantizationConfig]) -> None:
    for n in model.graph.nodes:
        if n.op != "call_function" or not _is_share_obs_or_fq_op(n.target):
            continue

        prev_node = n.args[0]
        if not isinstance(prev_node, Node):
            continue

        if prev_node.op == "placeholder":
            input_act_qspec = get_input_act_qspec(quantization_config)
            shared_qspec = SharedQuantizationSpec((prev_node, n))

            n.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map={
                    prev_node: input_act_qspec,
                },
                output_qspec=shared_qspec,
                _annotated=True,
            )

        else:
            quantization_annotation = prev_node.meta.get("quantization_annotation", None)
            if not quantization_annotation:
                continue

            output_qspec = quantization_annotation.output_qspec
            if not output_qspec:
                continue

            # make sure current node is not annotated
            if (
                "quantization_annotation" in n.meta
                and n.meta["quantization_annotation"]._annotated
            ):
                continue

            shared_qspec = SharedQuantizationSpec(prev_node)
            # propagate the previous output_qspec to the current node
            n.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map={
                    prev_node: shared_qspec,
                },
                output_qspec=shared_qspec,
                _annotated=True,
            )


# TODO: make the list of ops customizable
def _convert_scalars_to_attrs(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

    for n in model.graph.nodes:
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
        ]:
            continue
        args = list(n.args)
        new_args = []
        for i in range(len(args)):
            if isinstance(args[i], torch.fx.Node):
                new_args.append(args[i])
                continue
            prefix = "_tensor_constant_"
            get_new_attr_name = get_new_attr_name_with_prefix(prefix)
            tensor_constant_name = get_new_attr_name(model)
            float_tensor = torch.tensor(float(args[i])).to(device)
            model.register_buffer(tensor_constant_name, float_tensor)
            fake_mode = n.meta["val"].fake_mode
            with model.graph.inserting_before(n):
                get_attr_node = model.graph.create_node(
                    "get_attr", tensor_constant_name, (), {}
                )
                get_attr_node.meta["val"] = fake_mode.from_tensor(
                    float_tensor, static_shapes=True
                )
                new_args.append(get_attr_node)
        n.args = tuple(new_args)
    model.recompile()
    return model


def annotate_bias(model: torch.fx.GraphModule):
    for node in model.graph.nodes:
        if node.op == "call_function" and node.target in [
            torch.ops.aten.conv1d.default,
            torch.ops.aten.conv2d.default,
            # torch.ops.aten.linear.default,
        ]:
            if len(node.args) <= 2 or node.args[2] is None:
                continue
            input_act = node.args[0]
            assert isinstance(input_act, Node)
            weight = node.args[1]
            assert isinstance(weight, Node)
            bias = node.args[2]
            assert isinstance(bias, Node)

            def derive_qparams_fn(
                obs_or_fqs: list[ObserverOrFakeQuantize],
            ) -> tuple[Tensor, Tensor]:
                assert (
                    len(obs_or_fqs) == 2
                ), f"Expecting one weight obs/fq, got: {len(obs_or_fqs)}"
                act_obs_or_fq = obs_or_fqs[0]
                weight_obs_or_fq = obs_or_fqs[1]
                act_scale, act_zp = act_obs_or_fq.calculate_qparams()
                (
                    weight_scale,
                    weight_zp,
                ) = weight_obs_or_fq.calculate_qparams()
                return act_scale * weight_scale, weight_zp

            bias_qspec = DerivedQuantizationSpec(
                derived_from=[(input_act, node), (weight, node)],
                derive_qparams_fn=derive_qparams_fn,
                dtype=torch.int32,
                quant_min=-(2**31),
                quant_max=2**31 - 1,
                qscheme=torch.per_channel_symmetric,
                ch_axis=0,
            )
            node.meta["quantization_annotation"].input_qspec_map[bias] = bias_qspec