import os

import torch
import torch.nn.functional as F


BN_EPS = float(os.getenv("ULTRALYTICS_PT2E_BN_EPS", "1e-3"))
BN_MOMENTUM = float(os.getenv("ULTRALYTICS_PT2E_BN_MOMENTUM", "0.03"))


_PATCH_FLAG = "_ultralytics_pt2e_bn_patch_installed"
_QAT_PATCH_FLAG = "_ultralytics_pt2e_qat_bn_patch_installed"


def _toggle_exported_batch_norm_nodes(model: torch.fx.GraphModule, training: bool) -> None:
    """Only flip exported BN training flag and preserve momentum/eps literals."""
    model.graph.eliminate_dead_code()

    changed = False
    for node in model.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.batch_norm.default:
            continue
        if len(node.args) < 8:
            continue

        args = list(node.args)
        if args[5] == training:
            continue

        args[5] = training
        node.args = tuple(args)
        changed = True

    if changed:
        model.graph.eliminate_dead_code()
        model.recompile()


def patch_pt2e_batchnorm_handling() -> bool:
    """
    Monkey patch PT2E exported BN train/eval switching to preserve custom eps and momentum.

    Torch 2.6 `torch.ao.quantization.pt2e.export_utils._replace_batchnorm` rewrites exported
    batchnorm nodes with hard-coded default `momentum=0.1` and `eps=1e-5`. Ultralytics models
    initialize BN with `momentum=0.03` and `eps=1e-3`, so QAT/export validation can regress
    significantly after calling `move_exported_model_to_eval()` or `allow_exported_model_train_eval()`.
    """
    try:
        from torch.ao.quantization.pt2e import export_utils, qat_utils
        from torch.ao.quantization.pt2e.export_utils import _WrapperModule
    except Exception:
        return False

    changed = False

    if not getattr(export_utils, _PATCH_FLAG, False):
        def _replace_batchnorm_preserve_hyperparams(m: torch.fx.GraphModule, train_to_eval: bool) -> None:
            _toggle_exported_batch_norm_nodes(m, training=not train_to_eval)

        export_utils._replace_batchnorm = _replace_batchnorm_preserve_hyperparams
        setattr(export_utils, _PATCH_FLAG, True)
        changed = True

    if not getattr(qat_utils, _QAT_PATCH_FLAG, False):
        def _is_conv_transpose_fn(conv_fn) -> bool:
            return conv_fn in {F.conv_transpose1d, F.conv_transpose2d}

        def _get_qat_conv_bn_pattern_preserve_hparams(conv_fn):
            def _qat_conv_bn_pattern(
                x: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                bn_weight: torch.Tensor,
                bn_bias: torch.Tensor,
                bn_running_mean: torch.Tensor,
                bn_running_var: torch.Tensor,
            ) -> torch.Tensor:
                running_std = torch.sqrt(bn_running_var + BN_EPS)
                scale_factor = bn_weight / running_std
                weight_shape = [1] * len(conv_weight.shape)
                weight_in_channel_axis = 1 if _is_conv_transpose_fn(conv_fn) else 0
                weight_shape[weight_in_channel_axis] = -1
                bias_shape = [1] * len(conv_weight.shape)
                bias_shape[1] = -1
                scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
                zero_bias = torch.zeros_like(conv_bias, dtype=x.dtype)
                x = conv_fn(x, scaled_weight, zero_bias)
                x = x / scale_factor.reshape(bias_shape)
                x = x + conv_bias.reshape(bias_shape)
                x = F.batch_norm(
                    x,
                    bn_running_mean,
                    bn_running_var,
                    bn_weight,
                    bn_bias,
                    training=True,
                    momentum=BN_MOMENTUM,
                    eps=BN_EPS,
                )
                return x

            return _WrapperModule(_qat_conv_bn_pattern)

        def _get_qat_conv_bn_pattern_no_conv_bias_preserve_hparams(conv_fn):
            def _qat_conv_bn_pattern_no_conv_bias(
                x: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                bn_weight: torch.Tensor,
                bn_bias: torch.Tensor,
                bn_running_mean: torch.Tensor,
                bn_running_var: torch.Tensor,
            ) -> torch.Tensor:
                running_std = torch.sqrt(bn_running_var + BN_EPS)
                scale_factor = bn_weight / running_std
                weight_shape = [1] * len(conv_weight.shape)
                weight_in_channel_axis = 1 if _is_conv_transpose_fn(conv_fn) else 0
                weight_shape[weight_in_channel_axis] = -1
                bias_shape = [1] * len(conv_weight.shape)
                bias_shape[1] = -1
                scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
                x = conv_fn(x, scaled_weight, None)
                x = x / scale_factor.reshape(bias_shape)
                x = F.batch_norm(
                    x,
                    bn_running_mean,
                    bn_running_var,
                    bn_weight,
                    bn_bias,
                    training=True,
                    momentum=BN_MOMENTUM,
                    eps=BN_EPS,
                )
                return x

            return _WrapperModule(_qat_conv_bn_pattern_no_conv_bias)

        def _get_quantized_qat_conv_bn_pattern_preserve_hparams(
            is_per_channel: bool,
            has_bias: bool,
            bias_is_quantized: bool,
            conv_fn,
            bn_is_training: bool,
        ):
            def _quantized_qat_conv_bn_pattern(
                x: torch.Tensor,
                conv_weight: torch.Tensor,
                bn_weight: torch.Tensor,
                bn_bias: torch.Tensor,
                bn_running_mean: torch.Tensor,
                bn_running_var: torch.Tensor,
                **kwargs,
            ) -> torch.Tensor:
                running_std = torch.sqrt(bn_running_var + BN_EPS)
                scale_factor = bn_weight / running_std
                weight_shape = [1] * len(conv_weight.shape)
                weight_shape[0] = -1
                bias_shape = [1] * len(conv_weight.shape)
                bias_shape[1] = -1
                scaled_weight = conv_weight * scale_factor.reshape(weight_shape)
                scaled_weight = qat_utils._append_qdq(
                    scaled_weight,
                    is_per_channel,
                    is_bias=False,
                    kwargs=kwargs,
                )
                if has_bias:
                    zero_bias = torch.zeros_like(kwargs["conv_bias"], dtype=x.dtype)
                    if bias_is_quantized:
                        zero_bias = qat_utils._append_qdq(
                            zero_bias,
                            is_per_channel,
                            is_bias=True,
                            kwargs=kwargs,
                        )
                    x = conv_fn(x, scaled_weight, zero_bias)
                else:
                    x = conv_fn(x, scaled_weight, None)
                x = x / scale_factor.reshape(bias_shape)
                if has_bias:
                    x = x + kwargs["conv_bias"].reshape(bias_shape)
                x = F.batch_norm(
                    x,
                    bn_running_mean,
                    bn_running_var,
                    bn_weight,
                    bn_bias,
                    training=bn_is_training,
                    momentum=BN_MOMENTUM,
                    eps=BN_EPS,
                )
                return x

            return _WrapperModule(_quantized_qat_conv_bn_pattern)

        def _get_folded_quantized_qat_conv_bn_pattern_preserve_hparams(
            is_per_channel: bool,
            has_bias: bool,
            bias_is_quantized: bool,
            conv_fn,
            bn_is_training: bool,
        ):
            def _folded_quantized_qat_conv_bn_pattern(
                x: torch.Tensor,
                conv_weight: torch.Tensor,
                bn_weight: torch.Tensor,
                bn_bias: torch.Tensor,
                bn_running_mean: torch.Tensor,
                bn_running_var: torch.Tensor,
                **kwargs,
            ) -> torch.Tensor:
                conv_weight_local = qat_utils._append_qdq(
                    conv_weight,
                    is_per_channel,
                    is_bias=False,
                    kwargs=kwargs,
                )
                if has_bias:
                    bias = kwargs["conv_bias"]
                    if bias_is_quantized:
                        bias = qat_utils._append_qdq(
                            bias,
                            is_per_channel,
                            is_bias=True,
                            kwargs=kwargs,
                        )
                else:
                    bias = None
                x = conv_fn(x, conv_weight_local, bias)
                x = F.batch_norm(
                    x,
                    bn_running_mean,
                    bn_running_var,
                    bn_weight,
                    bn_bias,
                    training=bn_is_training,
                    momentum=BN_MOMENTUM,
                    eps=BN_EPS,
                )
                return x

            return _WrapperModule(_folded_quantized_qat_conv_bn_pattern)

        qat_utils._get_qat_conv_bn_pattern = _get_qat_conv_bn_pattern_preserve_hparams
        qat_utils._get_qat_conv_bn_pattern_no_conv_bias = _get_qat_conv_bn_pattern_no_conv_bias_preserve_hparams
        qat_utils._get_quantized_qat_conv_bn_pattern = _get_quantized_qat_conv_bn_pattern_preserve_hparams
        qat_utils._get_folded_quantized_qat_conv_bn_pattern = (
            _get_folded_quantized_qat_conv_bn_pattern_preserve_hparams
        )
        setattr(qat_utils, _QAT_PATCH_FLAG, True)
        changed = True

    return changed
