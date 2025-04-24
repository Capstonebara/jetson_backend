import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ssdlite_mobilenetv3 import ssdlite320_mobilenet_v3

class FaceDetectionModel:
    def __init__(self, model_path: str, device: str = None, detection_threshold: float = 0.99):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_threshold = detection_threshold
        self.model = ssdlite320_mobilenet_v3()
        checkpoint = torch.load(model_path, map_location=self.device)
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        self.model.load_state_dict(checkpoint)
        self.model.eval().to(self.device)
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def detection_preprocess(self, image):
        # Resize and normalize the image
        image = cv2.resize(image, (320, 320))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image
    
    def detect_faces(self, image):
        # Preprocess the image
        orig_height, orig_width = image.shape[:2]
        image = self.detection_preprocess(image)
        
        # Perform detection
        with torch.no_grad():
            detections = self.model(image)
            
        # Post-process detections
        boxes = detections[0][0]
        scores = detections[1][0]
        
        # Filter out low-confidence detections
        mask = scores >= self.detection_threshold
        filtered_boxes = boxes[mask]
        
        # Scale boxes back to original image size
        scale_x = orig_width / 320
        scale_y = orig_height / 320
        
        if len(filtered_boxes) > 0:
            scaled_boxes = torch.stack([
                filtered_boxes[:, 0] * scale_x,
                filtered_boxes[:, 1] * scale_y,
                filtered_boxes[:, 2] * scale_x,
                filtered_boxes[:, 3] * scale_y,
            ], dim=1).tolist()
            return scaled_boxes
        return []
        