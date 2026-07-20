"""版本兼容层:全包唯一的条件 import 块(迁移自 QAT.axera 的 qat-migrate-2_10 方案)。

torch == 2.10 → torchao 体系(torch.ao 的 PT2E 在 2.10 已坏);
其余版本      → torch.ao 体系(以 2.6 为基线)。严格按 major.minor 判定,
补丁版(如 2.10.1)视同 2.10。两套体系的符号改名差异在此对齐为 2.6 内部命名,
实现层零感知;get_aten_graph_module_for_pattern() 吃掉 2.6 的 using_training_ir
参数差异(2.10 已移除该 helper)。
"""
import warnings
import torch

_ver = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
IS_TORCH_210 = _ver == (2, 10)
BACKEND = "torchao" if IS_TORCH_210 else "torch.ao"

if _ver not in ((2, 6), (2, 10)):
    warnings.warn(
        f"utils 官方支持 torch 2.6 与 2.10(严格判定),当前 {torch.__version__} "
        f"未经验证,将按 torch.ao(2.6)路径处理——大概率不可用,请先完成版本验证",
        stacklevel=2,
    )

if IS_TORCH_210:
    from torchao.quantization.pt2e.quantize_pt2e import (  # noqa: F401
        prepare_qat_pt2e,
        convert_pt2e,
    )
    from torchao.quantization.pt2e import (  # noqa: F401
        move_exported_model_to_eval,
        move_exported_model_to_train,
        allow_exported_model_train_eval,
        disable_fake_quant,
        enable_fake_quant,
        disable_observer,
        enable_observer,
        observer,
        ObserverOrFakeQuantize,
        DerivedObserverOrFakeQuantize as _DerivedObserverOrFakeQuantize,
    )
    from torchao.quantization.pt2e.fake_quantize import (  # noqa: F401
        FakeQuantize,
        FusedMovingAvgObsFakeQuantize,
    )
    from torchao.quantization.pt2e.observer import (  # noqa: F401
        HistogramObserver,
        MinMaxObserver,
        MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver,
        PerChannelMinMaxObserver,
        PlaceholderObserver,
    )
    from torchao.quantization.pt2e.quantizer import (  # noqa: F401
        QuantizationAnnotation,
        QuantizationSpec,
        QuantizationSpecBase,
        SharedQuantizationSpec,
        DerivedQuantizationSpec,
        Quantizer,
    )
    from torchao.quantization.pt2e.quantizer.utils import (  # noqa: F401
        annotate_input_qspec_map as _annotate_input_qspec_map,
        annotate_output_qspec as _annotate_output_qspec,
        get_module_name_filter as _get_module_name_filter,
    )
    from torchao.quantization.pt2e.export_utils import (  # noqa: F401
        WrapperModule as _WrapperModule,
    )
    from torchao.quantization.pt2e.utils import (  # noqa: F401
        _get_aten_graph_module_for_pattern as _pattern_impl,
        _is_conv_node,
        _is_conv_transpose_node,
        get_new_attr_name_with_prefix,
    )

    def get_aten_graph_module_for_pattern(pattern, example_inputs, is_cuda, gm):
        return _pattern_impl(pattern, example_inputs, is_cuda)

else:
    from torch.ao.quantization.quantize_pt2e import (  # noqa: F401
        prepare_qat_pt2e,
        convert_pt2e,
    )
    from torch.ao.quantization import (  # noqa: F401
        move_exported_model_to_eval,
        move_exported_model_to_train,
        allow_exported_model_train_eval,
        disable_fake_quant,
        enable_fake_quant,
        disable_observer,
        enable_observer,
        observer,
        ObserverOrFakeQuantize,
        _DerivedObserverOrFakeQuantize,
    )
    from torch.ao.quantization.fake_quantize import (  # noqa: F401
        FakeQuantize,
        FusedMovingAvgObsFakeQuantize,
    )
    from torch.ao.quantization.observer import (  # noqa: F401
        HistogramObserver,
        MinMaxObserver,
        MovingAverageMinMaxObserver,
        MovingAveragePerChannelMinMaxObserver,
        PerChannelMinMaxObserver,
        PlaceholderObserver,
    )
    from torch.ao.quantization.quantizer import (  # noqa: F401
        QuantizationAnnotation,
        QuantizationSpec,
        QuantizationSpecBase,
        SharedQuantizationSpec,
        DerivedQuantizationSpec,
        Quantizer,
    )
    from torch.ao.quantization.quantizer.utils import (  # noqa: F401
        _annotate_input_qspec_map,
        _annotate_output_qspec,
        _get_module_name_filter,
    )
    from torch.ao.quantization.pt2e.export_utils import (  # noqa: F401
        _WrapperModule,
    )
    from torch.ao.quantization.pt2e.utils import (  # noqa: F401
        _get_aten_graph_module_for_pattern as _pattern_impl,
        _is_conv_node,
        _is_conv_transpose_node,
    )
    from torch.ao.quantization.fx.utils import (  # noqa: F401
        get_new_attr_name_with_prefix,
    )

    def get_aten_graph_module_for_pattern(pattern, example_inputs, is_cuda, gm):
        from torch._export import gm_using_training_ir

        return _pattern_impl(
            pattern, example_inputs, is_cuda,
            using_training_ir=gm_using_training_ir(gm),
        )


def capture_for_training(model, args, dynamic_shapes=None):
    """图捕获:2.6 用 export_for_training;2.10 用 export(export_for_training 在
    2.10 已弃用,官方称二者功能等价)。返回 ExportedProgram(.module() 取 GraphModule)。"""
    if IS_TORCH_210:
        return torch.export.export(model, args, dynamic_shapes=dynamic_shapes)
    return torch.export.export_for_training(model, args, dynamic_shapes=dynamic_shapes)
