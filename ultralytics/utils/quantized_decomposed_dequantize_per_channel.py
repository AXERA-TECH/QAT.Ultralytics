from typing import Optional
from onnxscript.onnx_opset import opset18 as op
from onnxscript.function_libs.torch_lib.ops import common
from onnxscript.function_libs.torch_lib.registration import torch_op
from onnxscript.onnx_types import TensorType


@torch_op(
    ("quantized_decomposed::dequantize_per_channel",),
    trace_only=True,
)
def quantized_decomposed_dequantize_per_channel(
    input: TensorType,
    scales: TensorType,
    zero_points: TensorType,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: int,
    out_dtype: Optional[int] = None,
) -> TensorType:
    # from IPython import embed; embed()
    zero_points = op.Cast(zero_points, to=dtype)
    dequantized = op.DequantizeLinear(
        input,
        scales,
        zero_points,
        axis=axis,
    )
    return dequantized


@torch_op(
    ("quantized_decomposed::quantize_per_channel",),
    trace_only=True,
)
def quantized_decomposed_quantize_per_channel(
    input: TensorType,
    scales: TensorType,
    zero_points: TensorType,
    axis: int,
    quant_min: int,
    quant_max: int,
    dtype: int,
    out_dtype: Optional[int] = None,
) -> TensorType:
    # from IPython import embed; embed()
    zero_points = op.Cast(zero_points, to=dtype)
    dequantized = op.QuantizeLinear(
        input,
        scales,
        zero_points,
        axis=axis,
    )
    return dequantized