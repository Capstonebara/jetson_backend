import os
import cv2
import json
import torch
import faiss
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms
from FaceDetectionModel import FaceDetectionModel
from FaceRecognitionModel import FaceRecognitionModel
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time face recognition pipeline using FAISS and PyTorch models"
    )
    parser.add_argument(
        "--detection_model_path", type=str, required=True,
        help="Path to the face detection model checkpoint"
    )
    parser.add_argument(
        "--embedding_model_path", type=str, required=True,
        help="Path to the face embedding (recognition) model checkpoint"
    )
    parser.add_argument(
        "--embedded_folder", type=str, required=True,
        help="Folder containing subfolders of JSON embeddings for known faces"
    )
    parser.add_argument(
        "--recognition_threshold", type=float, default=0.7,
        help="L2-distance threshold under which a face is considered recognized"
    )
    parser.add_argument(
        "--detection_threshold", type=float, default=0.99,
        help="Confidence threshold for face detection"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=512,
        help="Dimensionality of the face embedding vectors"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the models on ('cuda' or 'cpu')"
    )
    return parser.parse_args()

class FaceRecognitionPipeline:
    def __init__(
        self,
        device: str,
        recognition_threshold: float,
        detection_threshold: float,
        embedding_dim: int,
        detection_model_path: str,
        embedding_model_path: str,
        embedded_folder: str
    ):
        self.device = device
        self.recognition_threshold = recognition_threshold
        self.detection_threshold = detection_threshold
        
        # Detection model
        self.face_detector = FaceDetectionModel(model_path=detection_model_path, detection_threshold = self.detection_threshold, device=self.device)
        print("Using SSDLite320 MobileNet V3 for face detection")
        # Embedding model
        self.embedding_model = FaceRecognitionModel(model_path=embedding_model_path, device=self.device)
        print("Using KANFace model")
        
        # FAISS index
        self.embedding_dim = embedding_dim
        self.embedding_db = load_embedding_from_json(embedded_folder=embedded_folder)
        self.index, self.label_map = build_faiss_index(self.embedding_db, self.embedding_dim)
    
    def recognition(self, image):
        # Detect faces
        boxes = self.face_detector.detect_faces(image)
        
        results = []
        if boxes is not None:
            embeddings = []
            for i, box in enumerate(boxes):
                face = Image.fromarray(image).crop(box)
                
                embedding = self.embedding_model.get_embedding(face)
                embeddings.append(embedding[0])
            
            if embeddings:
                distances, indices = self.index.search(np.array(embeddings).astype('float32'), 1)
                for i, (distance, index) in enumerate(zip(distances, indices)):
                    print(f"Distance: {distance[0]}, Index: {index[0]}")
                    if distance[0] < self.recognition_threshold:
                        for label, (start, end) in self.label_map.items():
                            if start <= index[0] < end:
                                print(f"Recognized: {label}")
                                break
                    results.append((boxes[i], label, distance[0]))
        return results
    
    def realtime_recognition(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        while True:
            ret, frame = cap.read()
            # target_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                break
            
            results = self.recognition(frame)
            
            for box, label, distance in results:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {distance:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    args = parse_args()
    pipeline = FaceRecognitionPipeline(
        device=args.device,
        recognition_threshold=args.recognition_threshold,
        detection_threshold=args.detection_threshold,
        embedding_dim=args.embedding_dim,
        detection_model_path=args.detection_model_path,
        embedding_model_path=args.embedding_model_path,
        embedded_folder=args.embedded_folder
    )
    pipeline.realtime_recognition()