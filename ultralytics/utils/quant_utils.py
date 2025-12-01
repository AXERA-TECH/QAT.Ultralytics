import re
import onnx
import torch
import onnx_graphsurgeon as gs

from onnxslim import slim
from onnxruntime.quantization.quant_utils import pack_bytes_to_4bit


def simplify_and_fix_4bit_dtype(qat_path: str, sim_path: str):
    """
    1. 如果完全不做 constant falding,
    weight, scale, zero_point 会混乱地存储在 constant 算子或者 initializer 以及 cast 算子后面，
    不好处理
    2. 如果直接 slim,
    会把相同的 weight 合并, u4 和 u8 的 zero_point 如果都是 0 复用第一个，导致后面配置 4bit 时候分不开
    3. 如果把 slim 里的 constant falding 抽出来单独做
    后面再 slim 时候又会由于 onnx shape_inference 支持不足，导致 quant 即使是 4bit 的 zero_point,
    算子输出 value_info 还是会再被改回 8bit
    4. 最后搞成了下面先 constant falding 刷一遍 param, 再 slim 刷一遍 vi 的形式
    """
    # load
    onnx_model = onnx.load(qat_path)

    # 4bit info
    tensors_4bit = {}
    for node in onnx_model.graph.node:
        if node.op_type not in ["QuantizeLinear", "DequantizeLinear"]:
            continue

        metadata_props = {}
        for metadata_prop in node.metadata_props:
            metadata_props.update({metadata_prop.key: metadata_prop.value})
        namespace = metadata_props.get("namespace", None)
        fx_node = metadata_props.get("pkg.torch.onnx.fx_node", None)
        if not namespace or not fx_node:
            continue

        target = namespace.split(": ")[1]
        node_args = re.findall(r"args\s*=\s*\(([^)]+)\)", fx_node)[0].split(", ")

        if target == "quantized_decomposed.quantize_per_tensor.default":
            assert len(node_args) == 6
            input, scale, zp, quant_min, quant_max, dtype = node_args
            if int(quant_min) == 0 and int(quant_max) == 15 and dtype == "torch.uint8":
                tensors_4bit.update({node.output[0]: onnx.TensorProto.UINT4})
                tensors_4bit.update({node.input[2]: onnx.TensorProto.UINT4})
        elif target == "quantized_decomposed.dequantize_per_tensor.default":
            assert len(node_args) == 6
            input, scale, zp, quant_min, quant_max, dtype = node_args
            if int(quant_min) == 0 and int(quant_max) == 15 and dtype == "torch.uint8":
                tensors_4bit.update({node.input[0]: onnx.TensorProto.UINT4})
                tensors_4bit.update({node.input[2]: onnx.TensorProto.UINT4})
        elif target == "quantized_decomposed.dequantize_per_channel.default":
            assert len(node_args) == 7
            input, scale, zp, axis, quant_min, quant_max, dtype = node_args
            if int(quant_min) == -7 and int(quant_max) == 7 and dtype == "torch.int8":
                tensors_4bit.update({node.input[0]: onnx.TensorProto.INT4})
                tensors_4bit.update({node.input[2]: onnx.TensorProto.INT4})
        else:
            assert False, f"node target: [{target}] is illegal"

    # constant falding
    graph = gs.import_onnx(onnx_model).toposort()
    graph.fold_constants().cleanup().toposort()
    sim_model = gs.export_onnx(graph)

    # fix 4bit node
    vis = {}
    for vi in sim_model.graph.value_info:
        vis.update({vi.name: vi})
    params = {}
    for param in sim_model.graph.initializer:
        params.update({param.name: param})

    for name, dtype in tensors_4bit.items():
        if name in vis:
            vis[name].type.tensor_type.elem_type = dtype
        if name in params:
            data = onnx.numpy_helper.to_array(params[name])
            data = bytes(pack_bytes_to_4bit(data.tobytes()))
            params[name].data_type = dtype
            params[name].raw_data = data

    # sim
    sim_model = slim(sim_model)
    vis = {}
    for vi in sim_model.graph.value_info:
        vis.update({vi.name: vi})
    for name, dtype in tensors_4bit.items():
        if name in vis:
            vis[name].type.tensor_type.elem_type = dtype

    # save
    onnx.save(sim_model, sim_path)
    print(f"save onnx model to [{sim_path}] Successfully!")