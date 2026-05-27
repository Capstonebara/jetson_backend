#import cv2
import onnxruntime as ort
import numpy 
import torch

print("Available providers:", ort.get_available_providers())
ort_session = ort.InferenceSession("/home/jetson/FaceRecognitionSystem/jetson/backend/model/kanface/kanface.onnx",
                                   providers=["CUDAExecutionProvider"])

outputs = ort_session.run(
    None,
    # {"actual_input_1": np.random.randn(10, 3, 224, 224).astype(np.float32)},
    # {"images": image_tensor.numpy()}
    {"images": (torch.ones(1,3,112,112)).numpy()}
    # {torch.randn(1, 3, 320, 320)}
)

print(outputs[0])
