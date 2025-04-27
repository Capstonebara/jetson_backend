import os
import cv2
import json
import torch
import faiss
import numpy as np
import argparse
import time
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
        "--grid_size", type=int, required=True,
        help="Grid size for KANFace model"
    )
    parser.add_argument(
        "--rank_ratio", type=float, required=True,
        help="Rank ratio for KANFace model"
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
        "--detection_threshold", type=float, default=0.995,
        help="Confidence threshold for face detection"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=512,
        help="Dimensionality of the face embedding vectors"
    )
    parser.add_argument(
        "--frame_height", type=int, default=480,
        help="Height of the video frame"
    )
    parser.add_argument(
        "--frame_width", type=int, default=640,
        help="Width of the video frame"
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
        grid_size: int,
        rank_ratio: float,
        recognition_threshold: float,
        detection_threshold: float,
        embedding_dim: int,
        detection_model_path: str,
        embedding_model_path: str,
        embedded_folder: str,
        frame_height: int,
        frame_width: int
    ):
        self.device = device
        self.recognition_threshold = recognition_threshold
        self.detection_threshold = detection_threshold
        self.embedding_dim = embedding_dim
        self.frame_height = frame_height
        self.frame_width = frame_width
        
        # Detection model
        self.face_detector = FaceDetectionModel(model_path=detection_model_path, detection_threshold = self.detection_threshold, device=self.device)
        print("Using SSDLite320 MobileNet V3 for face detection")
        # Embedding model
        self.embedding_model = FaceRecognitionModel(model_path=embedding_model_path, embedding_dim=self.embedding_dim, grid_size=grid_size, rank_ratio=rank_ratio, device=self.device)
        print("Using KANFace model")
        
        # FAISS index
        self.embedding_db = load_embedding_from_json(embedded_folder=embedded_folder)
        self.index, self.label_map = build_faiss_index(self.embedding_db, self.embedding_dim)
    
    def recognition(self, image):
        # Detect faces
        boxes = self.face_detector.detect_faces(image)
        
        results = []
        if boxes is not None:
            embeddings = []
            valid_boxes = []
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, boxes[i])
                if (self.frame_height * self.frame_width / 2) >= (x2 - x1) * (y2 - y1) >= (self.frame_height * self.frame_width / 13):
                    face = Image.fromarray(image).crop(box)
                    
                    embedding = self.embedding_model.get_embedding(face)
                    embeddings.append(embedding[0]) 
                    valid_boxes.append((x1, y1, x2, y2))
            
            if embeddings:
                distances, indices = self.index.search(np.array(embeddings).astype('float32'), 1)
                for (dist,), idx, box in zip(distances, indices.flatten(), valid_boxes):
                    if dist < self.recognition_threshold:
                        # Find label by index range
                        for label, (start, end) in self.label_map.items():
                            if start <= idx < end:
                                print(f"Recognized: {label}")
                                break
                    else:
                        label = "Unknown"

                    results.append((box, label))   
        return results
    
    def realtime_recognition(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        elapsed_time = 0
        frame_count = 0
        fps = 0
        
        while True:
            ret, frame = cap.read()
            print(f"Frame dimensions: {self.frame_height} x {self.frame_width}")
            # target_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                break
            
            frame_count += 1
            
            start_time = time.time()
            results = self.recognition(frame)
            latency = time.time() - start_time
            elapsed_time += latency
            
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                elapsed_time = 0
                frame_count = 0
                
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(f"Latency: {latency:.2f}s")
            for box, label in results:
                x1, y1, x2, y2 = map(int, box)
                if label != "Unknown":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)
                    cv2.putText(frame, f"{label}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)
                    cv2.putText(frame, f"{label}", (x1, y1 - 10), 
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
        grid_size=args.grid_size,
        rank_ratio=args.rank_ratio,
        recognition_threshold=args.recognition_threshold,
        detection_threshold=args.detection_threshold,
        embedding_dim=args.embedding_dim,
        detection_model_path=args.detection_model_path,
        embedding_model_path=args.embedding_model_path,
        embedded_folder=args.embedded_folder,
        frame_width=args.frame_width,
        frame_height=args.frame_height
    )
    pipeline.realtime_recognition()