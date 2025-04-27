import torch
import numpy as np
import cv2
from torchvision import transforms
from kanface.models.KANFace import KANFace

class FaceRecognitionModel:
    def __init__(self, embedding_dim: int, grid_size: int, rank_ratio: float, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KANFace(num_features=embedding_dim, grid_size=grid_size, rank_ratio=rank_ratio, neuron_fun="mean")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval().to(self.device)
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
    def recognition_preprocess(self, image):
        # Resize and normalize the image
        image = cv2.resize(image, (112, 112))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image).unsqueeze(0).to(self.device)
        return image

    def get_embedding(self, image):
        # Preprocess the image
        # print(type(image))
        image = np.array(image)
        image = self.recognition_preprocess(image)
        
        # Get the embedding
        with torch.no_grad():
            embedding = self.model(image)
            embedding = embedding.cpu().numpy()
        return embedding
            
    