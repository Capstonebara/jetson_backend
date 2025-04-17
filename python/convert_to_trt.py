import torch
import torch_tensorrt
from typing import List, Optional, Dict
import torchvision  # This registers torchvision ops like nms
import cv2
from torchvision import transforms, models
# model = model.to("cuda")  # ✅ move model to GPU

# transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((320, 320)),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# img = cv2.imread("/home/jetson/FaceRecognitionSystem/jetson/backend/python/0_Parade_marchingband_1_5.jpg")
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_t = transform(img_rgb).unsqueeze(0).to('cuda')

# # Generate a dummy input and move to GPU
# input_tensor = torch.randn(1, 3, 320, 320).to("cuda")
#
# # Inference
# print(model(img_t))


# class FaceDetectionModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.detection.ssdlite320_mobilenet_v3_large(num_classes=2)
#
#     def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
#         if self.training:
#             outputs = self.model(images, targets)
#             return outputs[0]['bbox_regression'], outputs[0]['classification']
#         else:
#             outputs = self.model(images)
#             boxes = [out["boxes"] for out in outputs]
#             scores = [out["scores"] for out in outputs]
#             labels = [out["labels"] for out in outputs]
#             return boxes, scores, labels
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FaceDetectionModel()
#
# checkpoint = torch.load("./saved_model/face_detection3_epoch200_loss0.1802.pth")
# new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
# model.load_state_dict(new_state_dict)
# model.eval().to("cuda")
#
# dummy_input = [torch.randn(3, 320, 320).to("cuda")] 
#
# # Tracing
# with torch.no_grad():
#     traced_model = torch.jit.trace(model, (dummy_input,))
#
# # Save
# traced_model.save("saved_model/face_detection3_epoch200_loss0.1802_traced.ts")
# torch.save(model, "./saved_model/face_detection3_epoch200_loss0.1802.pt")

model = torch.jit.load('/home/jetson/FaceRecognitionSystem/jetson/backend/python/saved_model/kanface_06_25_128.ts')
model = model.eval().to("cuda")
# # print(type(model))

inputs=[
    torch_tensorrt.Input((1, 3, 112, 112))
]
#
enabled_precisions={torch.float16}  # Use FP16 for faster inference
# Compile with TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs = inputs,
    enabled_precisions=enabled_precisions,
    require_full_compilation = False # This is enable by default
)

input_data = [torch.randn(3, 112, 112).to("cuda").half()]

# result = trt_model(input_data)

print(f"Check legit : {torch.allclose(model(input_data), trt_model(input_data), rtol=1e-5, atol=1e-8)}")

# print("ok")
# Save the TensorRT optimized model
torch.jit.save(trt_model, "saved_model/trt_optimized_model.ts")

# class FaceDetectionModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = models.detection.ssdlite320_mobilenet_v3_large(num_classes=2)

#     def forward(self, x):
#         # Modified to accept a standard tensor instead of a list
#         # And handle both training and inference modes
#         if isinstance(x, torch.Tensor):
#             # Convert tensor to list with single tensor
#             images = [x]
#         else:
#             images = x
            
#         if self.training:
#             outputs = self.model(images, targets)
#             return outputs[0]['bbox_regression'], outputs[0]['classification']
#         else:
#             outputs = self.model(images)
#             boxes = [out["boxes"] for out in outputs]
#             scores = [out["scores"] for out in outputs]
#             labels = [out["labels"] for out in outputs]
#             return boxes, scores, labels

# # Create model and load weights
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FaceDetectionModel()

# checkpoint = torch.load("./saved_model/face_detection3_epoch200_loss0.1802.pth")
# new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
# model.load_state_dict(new_state_dict)
# model.eval().to("cuda")

# # Create dummy input with batch dimension
# dummy_input = torch.randn(1, 3, 320, 320).to("cuda").half()

# # Trace with the tensor input (not a list)
# with torch.no_grad():
#     traced_model = torch.jit.trace(model, dummy_input)

# # Save the correctly traced model
# traced_model.save("saved_model/face_detection_traced_tensor_input.ts")

# # Now compile with TensorRT
# trt_model = torch_tensorrt.compile(
#     traced_model,
#     inputs=[
#         torch_tensorrt.Input(
#             shape=(1, 3, 320, 320),
#             dtype=torch.float16
#         )
#     ],
#     enabled_precisions={torch.float16}
# )

# # Test inference
# test_input = torch.randn(1, 3, 320, 320).to("cuda").half()
# result = trt_model(test_input)
# print("TensorRT model inference successful!")

# # Save the TensorRT optimized model
# torch.jit.save(trt_model, "saved_model/trt_optimized_model.ts")
