import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from export import _align_split_reshape_quantization


def _make_split_reshape_model() -> onnx.ModelProto:
    initializers = [
        numpy_helper.from_array(np.array(0.1, dtype=np.float32), "split_scale"),
        numpy_helper.from_array(np.array(128, dtype=np.uint8), "split_zp"),
        numpy_helper.from_array(np.array(0.08, dtype=np.float32), "reshape_scale"),
        numpy_helper.from_array(np.array(128, dtype=np.uint8), "reshape_zp"),
        numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "reshape_shape"),
        numpy_helper.from_array(np.array([1, 1, 1], dtype=np.int64), "split_sizes"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "split_scale", "split_zp"], ["split_q"], name="split_q"),
        helper.make_node("DequantizeLinear", ["split_q", "split_scale", "split_zp"], ["split_dq"], name="split_dq"),
        helper.make_node(
            "Split",
            ["split_dq", "split_sizes"],
            ["split_0", "split_1", "split_2"],
            name="target_split",
            axis=1,
        ),
        helper.make_node("Reshape", ["split_2", "reshape_shape"], ["reshape_out"], name="target_reshape"),
        helper.make_node(
            "QuantizeLinear", ["reshape_out", "reshape_scale", "reshape_zp"], ["reshape_q"], name="reshape_q"
        ),
        helper.make_node(
            "DequantizeLinear", ["reshape_q", "reshape_scale", "reshape_zp"], ["reshape_dq"], name="reshape_dq"
        ),
        helper.make_node("QuantizeLinear", ["split_2", "reshape_scale", "reshape_zp"], ["other_q"], name="other_q"),
        helper.make_node("DequantizeLinear", ["other_q", "reshape_scale", "reshape_zp"], ["other_dq"], name="other_dq"),
    ]
    graph = helper.make_graph(
        nodes,
        "split_reshape_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
        [
            helper.make_tensor_value_info("reshape_dq", TensorProto.FLOAT, [1, 1]),
            helper.make_tensor_value_info("other_dq", TensorProto.FLOAT, [1, 1]),
        ],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def test_align_split_reshape_quantization_is_safe_and_idempotent():
    model = _make_split_reshape_model()
    original_initializer_count = len(model.graph.initializer)

    assert _align_split_reshape_quantization(model) == 1
    onnx.checker.check_model(model)

    nodes = {node.name: node for node in model.graph.node}
    initializers = {initializer.name: numpy_helper.to_array(initializer) for initializer in model.graph.initializer}
    reshape_q = nodes["reshape_q"]
    reshape_dq = nodes["reshape_dq"]
    other_q = nodes["other_q"]

    assert reshape_q.input[1:] == reshape_dq.input[1:]
    assert reshape_q.input[1] != "reshape_scale"
    assert reshape_q.input[2] != "reshape_zp"
    assert np.array_equal(initializers[reshape_q.input[1]], initializers["split_scale"])
    assert np.array_equal(initializers[reshape_q.input[2]], initializers["split_zp"])
    assert other_q.input[1:] == ["reshape_scale", "reshape_zp"]
    assert np.isclose(float(initializers["reshape_scale"]), 0.08)
    assert int(initializers["reshape_zp"]) == 128
    assert len(model.graph.initializer) == original_initializer_count + 2

    assert _align_split_reshape_quantization(model) == 0
    assert len(model.graph.initializer) == original_initializer_count + 2


def test_align_split_reshape_quantization_skips_uncovered_range():
    model = _make_split_reshape_model()
    reshape_scale = next(item for item in model.graph.initializer if item.name == "reshape_scale")
    reshape_scale.CopyFrom(numpy_helper.from_array(np.array(0.2, dtype=np.float32), "reshape_scale"))

    assert _align_split_reshape_quantization(model) == 0
    nodes = {node.name: node for node in model.graph.node}
    assert nodes["reshape_q"].input[1:] == ["reshape_scale", "reshape_zp"]
