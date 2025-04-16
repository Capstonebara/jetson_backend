import torch
from torchvision import models

# class FaceDetectionModel(torch.nn.Module):
#     def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
#         super().__init__()
#         self.device = device
#         self.model = models.detection.ssdlite320_mobilenet_v3_large(pretrained=False, num_classes=2)
#         self.model.to(device)
#         self.model.eval()

#     def forward(self, images, targets=None):
#         if self.training:
#             outputs = self.model(images, targets)
#             return outputs
#         else:
#             return self.model(images)

#     def load_weights(self, weights_path):
#         try:
#             # Load the state dict
#             checkpoint = torch.load(weights_path, map_location=self.device)
            
#             # Create a new state dict with corrected keys
#             new_state_dict = {}
#             for k, v in checkpoint.items():
#                 # Handle different prefix patterns that might be in the weights
#                 if k.startswith('module.model.'):
#                     new_key = k[13:]  # Remove 'module.model.' prefix
#                     new_state_dict[new_key] = v
#                 elif k.startswith('model.'):
#                     new_key = k[6:]  # Remove 'model.' prefix
#                     new_state_dict[new_key] = v
#                 elif k.startswith('module.'):
#                     new_key = k[7:]  # Remove 'module.' prefix
#                     new_state_dict[new_key] = v
#                 else:
#                     new_state_dict[k] = v
                
#             # Load with strict=False to allow partial matches
#             self.model.load_state_dict(new_state_dict, strict=False)
#             print("Custom face detection weights loaded successfully!")
#         except Exception as e:
#             # More detailed error without falling back to pretrained
#             print(f"ERROR: Failed to load weights from {weights_path}: {e}")
#             print("Check that the weights file exists and has the correct format.")
#             print("Paths are relative to where you run the script, not where the script is located.")
#             raise e  # Re-raise the exception to stop execution

#     def detect_faces(self, image):
#         """Detect faces in an image"""
#         # Convert numpy array to tensor
#         if isinstance(image, torch.Tensor):
#             img_tensor = image
#         else:
#             # Convert numpy array to tensor
#             img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
            
#         # Add batch dimension
#         img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
#         # Get model predictions
#         with torch.no_grad():
#             detections = self.model(img_tensor)
            
#         # Process results
#         boxes = detections[0]['boxes'].cpu().numpy()
#         scores = detections[0]['scores'].cpu().numpy()
        
#         # Filter detections with score > 0.5
#         face_boxes = []
#         for box, score in zip(boxes, scores):
#             if score > 0.5:
#                 face_boxes.append(box.astype(int))
                
#         return face_boxes if face_boxes else None


class FaceDetectionModel(torch.nn.Module):
    def __init__(self, device="cuda"):
        super(FaceDetectionModel, self).__init__()
        self.device = device
        # Load your model here
        self.model = models.detection.ssdlite320_mobilenet_v3_large(num_classes=2)
        # self.model = torch.load("./detection_model/face_detection3_epoch200_loss0.1802.pth", map_location=self.device)

    def forward(self, images, targets=None):
        if self.training:
            outputs = self.model(images, targets)
            return outputs["bbox_regression"], outputs["classification"]
        else:
            outputs = self.model(images)

            boxes = [out["boxes"] for out in outputs]
            scores = [out["scores"] for out in outputs]
            labels = [out["labels"] for out in outputs]

            return boxes, scores, labels