import re
import onnx
import torch
import onnx_graphsurgeon as gs

from onnxslim import slim
from onnxruntime.quantization.quant_utils import pack_bytes_to_4bit
from ._compat import _DerivedObserverOrFakeQuantize
from ._compat import HistogramObserver


def _castlike_to_cast(onnx_model):
    """CastLike(x, like) 的 like 是运行时张量时无法被常量折叠(FP32 区域 conv 的
    零 bias 会以 Expand(CastLike(标量)) 形式残留);本管线中 like 一律是 float
    激活,借 shape_inference 推断目标类型后改写成显式 Cast,让后续
    fold_constants 能把整条链折成常量(与 2.6 optimize() 的产物对齐)。"""
    if not any(n.op_type == "CastLike" for n in onnx_model.graph.node):
        return
    inferred = onnx.shape_inference.infer_shapes(onnx_model)
    vi_dtype = {v.name: v.type.tensor_type.elem_type
                for v in list(inferred.graph.value_info)
                + list(inferred.graph.input) + list(inferred.graph.output)}
    for node in onnx_model.graph.node:
        if node.op_type != "CastLike":
            continue
        target = vi_dtype.get(node.input[1], onnx.TensorProto.FLOAT)
        node.op_type = "Cast"
        del node.input[1]
        node.attribute.append(onnx.helper.make_attribute("to", target))


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
    _castlike_to_cast(onnx_model)

    # 4bit info
    tensors_4bit = {}
    for node in onnx_model.graph.node:
        if node.op_type not in ["QuantizeLinear", "DequantizeLinear"]:
            continue

        metadata_props = {}
        for metadata_prop in node.metadata_props:
            metadata_props.update({metadata_prop.key: metadata_prop.value})
        fx_node = metadata_props.get("pkg.torch.onnx.fx_node", None)
        if not fx_node:
            continue

        # 双格式兼容:2.10 的 namespace 格式已变(…Wrapper/quantize_per_tensor),
        # 优先从 fx_node 提取完整 target;2.6 老格式回退 namespace 第二段
        target_match = re.search(r"target=torch\.ops\.(quantized_decomposed\.\w+\.\w+)", fx_node)
        if target_match:
            target = target_match.group(1)
        else:
            namespace = metadata_props.get("namespace", "")
            target = namespace.split(": ")[1] if ": " in namespace else ""
            if not target.startswith("quantized_decomposed."):
                continue
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
    # gs/slim 的 make_model 会按当前 onnx 库默认值盖章 ir_version(1.19 时代=12),
    # ORT 1.23(max 11)与老 pulsar2 会拒载;回写成 dynamo 导出的 10
    sim_model.ir_version = 10
    onnx.save(sim_model, sim_path)
    print(f"save onnx model to [{sim_path}] Successfully!")


def load_ptq_calibration_to_qat(prepared_model_ptq: torch.fx.GraphModule, prepared_model_qat: torch.fx.GraphModule):
    assert len(prepared_model_qat.graph.nodes) == len(prepared_model_ptq.graph.nodes)
    with torch.no_grad():
        for node_ptq, node_qat in zip(prepared_model_ptq.graph.nodes, prepared_model_qat.graph.nodes):
            if node_ptq.op == "call_module" and node_ptq.target.startswith("activation_post_process"):
                assert node_qat.op == "call_module" and node_qat.target.startswith("activation_post_process")
                assert node_ptq.target == node_qat.target

                module_ptq = prepared_model_ptq.get_submodule(node_ptq.target)
                module_qat = prepared_model_qat.get_submodule(node_qat.target)

                if isinstance(module_ptq, _DerivedObserverOrFakeQuantize):
                    continue

                scale, zero_point = module_ptq.calculate_qparams()
                min_val = module_ptq.min_val
                max_val = module_ptq.max_val
                if isinstance(module_ptq, HistogramObserver):
                    min_val, max_val = module_ptq._non_linear_param_search()

                module_qat.scale = scale
                module_qat.zero_point = zero_point.type(torch.int32)
                module_qat.activation_post_process.min_val = min_val
                module_qat.activation_post_process.max_val = max_val