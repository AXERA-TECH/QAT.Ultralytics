import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from scripts.fix_requant_qparams import align_requant_qparams


def _make_requant_model() -> onnx.ModelProto:
    initializers = [
        numpy_helper.from_array(np.array(0.2, dtype=np.float32), "up_scale"),
        numpy_helper.from_array(np.array(10, dtype=np.uint8), "up_zp"),
        numpy_helper.from_array(np.array(0.1, dtype=np.float32), "down_scale"),
        numpy_helper.from_array(np.array(11, dtype=np.uint8), "down_zp"),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "up_scale", "up_zp"], ["up_q"], name="up_q"),
        helper.make_node("DequantizeLinear", ["up_q", "up_scale", "up_zp"], ["up_dq"], name="up_dq"),
        helper.make_node("Identity", ["up_dq"], ["requant_out"], name="requant_marker"),
        helper.make_node(
            "QuantizeLinear", ["requant_out", "down_scale", "down_zp"], ["down_q"], name="down_q"
        ),
        helper.make_node(
            "DequantizeLinear", ["down_q", "down_scale", "down_zp"], ["output"], name="down_dq"
        ),
        # This unrelated branch verifies that shared source initializers remain untouched.
        helper.make_node(
            "QuantizeLinear", ["other_input", "down_scale", "down_zp"], ["other_q"], name="other_q"
        ),
        helper.make_node(
            "DequantizeLinear", ["other_q", "down_scale", "down_zp"], ["other_output"], name="other_dq"
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "requant_test",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("other_input", TensorProto.FLOAT, [1]),
        ],
        [
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("other_output", TensorProto.FLOAT, [1]),
        ],
        initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


def test_align_requant_qparams_selects_larger_range_and_is_idempotent():
    model = _make_requant_model()

    alignments, skipped = align_requant_qparams(model)

    assert not skipped
    assert len(alignments) == 1
    assert alignments[0].selected_from == "up_q"
    assert np.isclose(alignments[0].scale, 0.2)
    assert alignments[0].zero_point == 10

    nodes = {node.name: node for node in model.graph.node}
    aligned_inputs = {tuple(nodes[name].input[1:3]) for name in ("up_q", "up_dq", "down_q", "down_dq")}
    assert len(aligned_inputs) == 1
    scale_name, zero_point_name = aligned_inputs.pop()
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    assert np.isclose(float(initializers[scale_name]), 0.2)
    assert int(initializers[zero_point_name]) == 10
    assert nodes["other_q"].input[1:3] == ["down_scale", "down_zp"]
    assert np.isclose(float(initializers["down_scale"]), 0.1)
    onnx.checker.check_model(model)

    second_alignments, second_skipped = align_requant_qparams(model)
    assert not second_alignments
    assert not second_skipped


def test_align_requant_qparams_can_select_downstream_range():
    model = _make_requant_model()
    initializers = {item.name: item for item in model.graph.initializer}
    initializers["up_scale"].CopyFrom(numpy_helper.from_array(np.array(0.05, dtype=np.float32), "up_scale"))

    alignments, skipped = align_requant_qparams(model, {"requant_marker"})

    assert not skipped
    assert len(alignments) == 1
    assert alignments[0].selected_from == "down_q"
    assert np.isclose(alignments[0].scale, 0.1)
    assert alignments[0].zero_point == 11


def test_align_requant_qparams_skips_non_scalar_parameters():
    model = _make_requant_model()
    initializers = {item.name: item for item in model.graph.initializer}
    initializers["up_scale"].CopyFrom(
        numpy_helper.from_array(np.array([0.2, 0.3], dtype=np.float32), "up_scale")
    )

    alignments, skipped = align_requant_qparams(model)

    assert not alignments
    assert skipped == [("requant_marker", "Q/DQ 参数必须是 initializer 中的有效标量 scale 和 zero-point")]
