import onnx

input_path = "./yolo26_onnx/yolo26n.onnx"
output_path = "./onnx/yolo26n-backbone.onnx"

input_names = ["images"]
output_names = [
    "/model.23/Reshape_1_output_0",
    "/model.23/Reshape_3_output_0",
    "/model.23/Reshape_5_output_0",
    "/model.23/Reshape_output_0",
    "/model.23/Reshape_2_output_0",
    "/model.23/Reshape_4_output_0",
]
onnx.utils.extract_model(input_path, output_path, input_names, output_names)
