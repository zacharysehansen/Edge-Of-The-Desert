import onnx

m = onnx.load("model/grace.onnx")
print(m.opset_import)
