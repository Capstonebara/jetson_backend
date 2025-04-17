import cv2
import numpy as np
import onnxruntime

session = onnxruntime.InferenceSession(
    "./saved_model/model.onnx",
    providers=['CUDAExecutionProvider']
)