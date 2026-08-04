import copy

import pytest
import torch
from torch import nn
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from torch.export import Dim

from ultralytics.nn.modules.block import SPPF
from ultralytics.utils.pt2e_bn_patch import patch_pt2e_batchnorm_handling


def _get_batch_norm_node_args(graph_module: torch.fx.GraphModule):
    for node in graph_module.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.batch_norm.default:
            return tuple(node.args)
    raise RuntimeError("No aten.batch_norm.default node found in exported graph")


class _BatchNormOnlyModel(nn.Module):
    def __init__(self, eps: float = 1e-3, momentum: float = 0.03):
        super().__init__()
        self.bn = nn.BatchNorm2d(4, eps=eps, momentum=momentum)

    def forward(self, x):
        return self.bn(x)


class _ConvBnReluModel(nn.Module):
    def __init__(self, eps: float = 1e-3, momentum: float = 0.03):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4, eps=eps, momentum=momentum)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


@pytest.mark.skipif(not hasattr(torch, "export"), reason="PT2E export requires torch.export")
def test_pt2e_export_eval_preserves_bn_hparams():
    patch_pt2e_batchnorm_handling()
    model = _BatchNormOnlyModel().train()
    inputs = (torch.randn(2, 4, 8, 8),)

    exported = torch.export.export_for_training(copy.deepcopy(model), inputs).module()
    before = _get_batch_norm_node_args(exported)

    torch.ao.quantization.allow_exported_model_train_eval(exported)
    exported.eval()
    after = _get_batch_norm_node_args(exported)

    assert before[6] == pytest.approx(model.bn.momentum)
    assert before[7] == pytest.approx(model.bn.eps)
    assert after[6] == pytest.approx(model.bn.momentum)
    assert after[7] == pytest.approx(model.bn.eps)
    assert after[5] is False


@pytest.mark.skipif(not hasattr(torch, "export"), reason="PT2E export requires torch.export")
def test_prepare_qat_preserves_bn_hparams_and_alignment():
    patch_pt2e_batchnorm_handling()
    model = _ConvBnReluModel().train()
    inputs = (torch.randn(2, 4, 8, 8),)

    exported = torch.export.export_for_training(copy.deepcopy(model), inputs).module()
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_qat=True))
    prepared = prepare_qat_pt2e(exported, quantizer)

    prepared_bn = _get_batch_norm_node_args(prepared)
    torch.ao.quantization.allow_exported_model_train_eval(prepared)
    prepared.apply(torch.ao.quantization.disable_observer)
    prepared.apply(torch.ao.quantization.disable_fake_quant)
    prepared.eval()
    prepared_eval_bn = _get_batch_norm_node_args(prepared)

    reference = torch.export.export_for_training(copy.deepcopy(model), inputs).module()
    torch.ao.quantization.allow_exported_model_train_eval(reference)
    reference.eval()

    with torch.no_grad():
        reference_out = reference(inputs[0])
        prepared_out = prepared(inputs[0])

    max_abs_diff = float((reference_out - prepared_out).abs().max())

    assert prepared_bn[6] == pytest.approx(model.bn.momentum)
    assert prepared_bn[7] == pytest.approx(model.bn.eps)
    assert prepared_eval_bn[6] == pytest.approx(model.bn.momentum)
    assert prepared_eval_bn[7] == pytest.approx(model.bn.eps)
    assert max_abs_diff < 1e-6


@pytest.mark.skipif(not hasattr(torch, "export"), reason="PT2E export requires torch.export")
def test_sppf_export_for_training_preserves_recursive_pooling():
    model = SPPF(256, 256).train()
    example = (torch.randn(2, 256, 20, 20),)
    exported = torch.export.export_for_training(
        copy.deepcopy(model),
        example,
        dynamic_shapes={"x": {0: Dim("batch", min=1, max=8), 2: 20, 3: 20}},
    ).module()
    probe = torch.randn(1, 256, 20, 20)

    with torch.no_grad():
        eager_out = model(probe)
        exported_out = exported(probe)

    max_abs_diff = float((eager_out - exported_out).abs().max())
    assert max_abs_diff < 1e-6
