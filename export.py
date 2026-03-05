from ultralytics import YOLO
# export onnx
import onnx
from onnxsim import simplify
from onnxslim import slim

import copy
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.ao.quantization.quantize_pt2e import (
    prepare_qat_pt2e,
    convert_pt2e,
)

from ultralytics.utils.ax_quantizer import(
    load_config,
    AXQuantizer,
)

from ultralytics.utils.quant_utils import simplify_and_fix_4bit_dtype
import ultralytics.utils.quantized_decomposed_dequantize_per_channel

import torch

#---Load a model---
model = YOLO("yolo11s.yaml")

#---export config---
qat_onnx_imgsz = [640, 640] # 推理模型输入大小
device = 'cuda'             # 
qat_onnx_sp = './qat.onnx'  # 保存路径，最终会导出qat_slim.onnx
qat_weights = 'runs/detect/train3/weights/epoch1.pt'    # qat权重
qat_weight_dict_sp = './qat.pth' # 保存qat权重路径 

#---quantizer config---
global_config, regional_configs = load_config("./config.json")
quantizer = AXQuantizer()
quantizer.set_global(global_config)
quantizer.set_regional(regional_configs)

#---export training model---
float_model = model.model.to(device)
inputs = torch.rand(1, 3, *qat_onnx_imgsz).to(device)
dynamic_shapes = None
print('start export!')
exported_model = torch.export.export_for_training(float_model, (inputs,), dynamic_shapes=dynamic_shapes).module()
print('export training model done!')

#---export quantized model---
prepared_model = prepare_qat_pt2e(exported_model, quantizer)
print('prepared model done!') 
torch.ao.quantization.move_exported_model_to_eval(prepared_model)
torch.ao.quantization.allow_exported_model_train_eval(prepared_model)
# print(f'prepared_model {prepared_model}')

#---load and save qat weights---
qat_weight_dict = torch.load(qat_weights)['qat_model']
torch.save(qat_weight_dict, qat_weight_dict_sp)
prepared_model.load_state_dict(qat_weight_dict)
print('load_state_dict done!')

#---export quantized model to onnx---
quantized_model = convert_pt2e(prepared_model)
print('convert_pt2e done!')
onnx_program = torch.onnx.export(quantized_model, (inputs.to(device),), dynamo=True, opset_version=21)
onnx_program.optimize()
onnx_program.save(qat_onnx_sp)
print(f'export qat model to [{qat_onnx_sp}] done!')

model_simp = slim(onnx_program.model_proto)
sim_path = qat_onnx_sp.replace('.onnx', '_slim.onnx')
onnx.save(model_simp, sim_path)
print(f"save onnx model to [{sim_path}] Successfully!")
