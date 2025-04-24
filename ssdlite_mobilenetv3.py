import torch
import torchvision
from torchvision import models

class ssdlite320_mobilenet_v3(torch.nn.Module):
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