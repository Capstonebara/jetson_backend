import torch
import torchvision
from torchvision import models
from torch2trt import torch2trt

class FaceDetectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.detection.ssdlite320_mobilenet_v3_large(num_classes=2)

    def forward(self, images, targets=None):
        if self.training:
            outputs = self.model(images, targets)

            return outputs['bbox_regression'], outputs['classification']
        else:
            outputs = self.model(images) 

            boxes = [out["boxes"] for out in outputs]
            scores = [out["scores"] for out in outputs]
            labels = [out["labels"] for out in outputs]

            return boxes, scores, labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceDetectionModel()

checkpoint = torch.load("./saved_model/face_detection3_epoch200_loss0.1802.pth")
new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
model.load_state_dict(new_state_dict)

model = model.eval().to(device)

x = torch.rand((1, 3, 320, 320)).to(device)
model_trt = torch2trt(model, [x])
#
# y = model(x)
# y_trt = model_trt(x)
#
# # check the output against PyTorch
# print(torch.max(torch.abs(y - y_trt)))
#
# torch.save(model_trt.state_dict(), './saved_model/ssd_trt.pth')
# import torch
# from torch2trt import torch2trt
# from torchvision.models.alexnet import alexnet

# create some regular pytorch model...
# model = alexnet(pretrained=True).eval().cuda()

# create example data
# x = torch.ones((1, 3, 224, 224)).cuda()

# convert to TensorRT feeding sample data as input
# model_trt = torch2trt(model, [x])

# y = model(x)
# y_trt = model_trt(x)

# check the output against PyTorch
# print(torch.max(torch.abs(y - y_trt)))
# torch.save(model_trt.state_dict(), 'alexnet_trt.pth')

# from torch2trt import TRTModule
#
# model_trt = TRTModule()
#
# model_trt.load_state_dict(torch.load('alexnet_trt.pth'))
