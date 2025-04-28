import cv2 as cv2
import torch
import torchvision
import numpy as np

img = cv2.imread("/home/jetson/FaceRecognitionSystem/jetson/backend/assets/Members.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).contiguous().unsqueeze(0)
print(tensor.dtype)
print(tensor.device)
print(tensor.shape)
print(tensor.is_contiguous())
print(tensor[:, 2, :5, :5])
print(tensor[:, 2, :5, :5].unsqueeze(0).shape)

tensor.cpu().numpy().astype(np.float32).tofile("python_tensor.bin")


# model = torch.jit.load("/home/jetson/FaceRecognitionSystem/jetson/backend/model/ssdlite/ssdlite_320_trace.ts")
# model.eval()
# device = next(model.parameters()).device
# print(f"model device {device}")
# with torch.no_grad():
#     output = model(tensor.to("cuda"))

# print(output)

# load_and_save.py
# import cv2
# import numpy as np
#
# # Load as BGR (default)
# img = cv2.imread("/home/jetson/FaceRecognitionSystem/jetson/backend/assets/LossOverEpochs.png", cv2.IMREAD_COLOR)
#
# # Save raw pixel data to binary file
# img.astype(np.uint8).tofile("loaded_python_bgr.bin")
#
# # Save shape for reference
# print("Python shape:", img.shape)  # (H, W, 3)

# with open("loaded_python_bgr.bin", "rb") as f1, open("loaded_cpp_bgr.bin", "rb") as f2:
#     data1 = f1.read()
#     data2 = f2.read()
#
# if data1 == data2:
#     print("✅ Loaded images are EXACTLY the same!")
# else:
#     print("❌ Loaded images are DIFFERENT!")
#     for i, (b1, b2) in enumerate(zip(data1, data2)):
#         if b1 != b2:
#             print(f"First mismatch at byte {i}: Python={b1}, C++={b2}")
#             break

with open("python_tensor.bin", "rb") as f1, open("cpp_tensor.bin", "rb") as f2:
    b1 = f1.read()
    b2 = f2.read()

    if b1 == b2:
        print("✅ Tensors are EXACTLY the same!")
    else:
        print("❌ Tensors are DIFFERENT!")
        for i in range(min(len(b1), len(b2))):
            if b1[i] != b2[i]:
                print(f"Mismatch at byte {i}: Python={b1[i]}, C++={b2[i]}")
                break
