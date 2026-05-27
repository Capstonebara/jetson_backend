import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, ops, transforms
from tqdm import tqdm
import cv2
import numpy as np
import time
import argparse
import os

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    ) 

# Define the model
class FaceDetectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.detection.ssdlite320_mobilenet_v3_large(num_classes=2)

    def forward(self, images, targets=None):
        return self.model(images, targets) if self.training else self.model(images)

# Load trained model
def load_model(model_path):
    model = FaceDetectionModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.eval()
    return model

# Process frame for video/webcam with optimized pipeline
def process_frame(model, frame):
    orig_height, orig_width = frame.shape[:2]
    # Resize frame using cv2.resize (faster than torchvision transforms)
    resized_frame = cv2.resize(frame, (320, 320))
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # Chuyển đổi ảnh sang tensor: (H, W, C) -> (C, H, W), chuẩn hóa từ [0, 255] về [0, 1]
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        detections = model(img_tensor)[0]

    # Scale bounding boxes về kích thước gốc của ảnh
    scale = torch.tensor([orig_width / 320, orig_height / 320, orig_width / 320, orig_height / 320]).to(device)
    detections['boxes'] = detections['boxes'] * scale

    faces_detected = int((detections['scores'] > 0.8).sum().item())
    return faces_detected, detections

# Global list for tính FPS trung bình
fps_list = []

# Inference function with optimized processing
def run_inference(model, source_type, source, output_path):
    total_faces = 0
    start_time = time.time()
    frame_count = 0

    if source_type == "image":
        img = cv2.imread(source)
        if img is None:
            print("Không đọc được ảnh từ đường dẫn đã cho!")
            return
        orig_height, orig_width = img.shape[:2]
        
        # Xử lý ảnh với pipeline tối ưu
        img_rgb = cv2.cvtColor(cv2.resize(img, (320, 320)), cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            detections = model(img_tensor)[0]
        
        scale = torch.tensor([orig_width / 320, orig_height / 320, orig_width / 320, orig_height / 320]).to(device)
        detections['boxes'] = detections['boxes'] * scale
        total_faces = int((detections['scores'] > 0.8).sum().item())
        visualize_results(img, detections, None, output_path)
        
    elif source_type == "video":
        cap = cv2.VideoCapture(source)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed to mp4v
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            faces_detected, detections = process_frame(model, frame)
            frame_time = time.time() - frame_start
            frame_fps = 1.0 / frame_time if frame_time > 0 else 0
            
            total_faces += faces_detected
            visualize_results(frame, detections, frame_fps, None)
            out.write(frame)
            
            frame_count += 1
        
        cap.release()
        out.release()
    
        total_time = time.time() - start_time
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        print(f"Total Faces Detected: {total_faces}, Processing Time: {total_time:.2f}s, Avg FPS: {avg_fps:.2f}")

    elif source_type == "webcam":
        cap = cv2.VideoCapture((gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER) if source_type == "webcam" else source)
        if not cap.isOpened():
            print("Không mở được nguồn video/webcam!")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            faces_detected, detections = process_frame(model, frame)
            frame_time = time.time() - frame_start
            frame_fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_list.append(frame_fps)
            total_faces += faces_detected
            visualize_results(frame, detections, frame_fps, None)
            
            # Hiển thị khung hình trên màn hình
            cv2.imshow("Video/Webcam Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"Total Faces Detected: {total_faces}, Processing Time: {total_time:.2f}s, Avg FPS: {avg_fps:.2f}")

# Visualization function remains unchanged
def visualize_results(img, detections, fps, output_path):
    if detections:
        for box, score in zip(detections['boxes'], detections['scores']):
            if score > 0.8:
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if fps is not None:
        cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    if output_path and not output_path.endswith(".mp4"):
        cv2.imwrite(output_path, img)

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--source_type", type=str, choices=["image", "video", "webcam"], required=True, help="Type of input source")
    parser.add_argument("--source", type=str, help="Path to image or video file (leave empty for webcam)")
    parser.add_argument("--output", type=str, required=True, help="Path to save output video or image")
    
    args = parser.parse_args()
    
    model = load_model(args.model_path)
    run_inference(model, args.source_type, args.source, args.output)
