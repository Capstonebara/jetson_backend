#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <torch/script.h>
#include <torch/torch.h>
#include <torchvision/vision.h>

std::string gstreamer_pipeline(int sensor_id, int width, int height) {
    std::ostringstream pipeline;
    pipeline << "nvarguscamerasrc sensor-id=" << sensor_id << " ! "
             << "video/x-raw(memory:NVMM), width=" << width << ", height=" << height << ", framerate=(fraction)60/1 ! "
             << "nvvidconv flip-method=0 ! "
             << "video/x-raw(memory:NVMM), format=(string)NV12 ! "
             << "nvvidconv ! video/x-raw, format=(string)BGRx ! "
             << "videoconvert ! video/x-raw, format=(string)BGR ! "
             << "appsink sync=false drop=true max-buffers=2";
    return pipeline.str();
}

int main(int argc, char* argv[]) {
    // Default values
    int camera_id = 0;
    int width = 640;  // Start with lower resolution
    int height = 480;
    int frame_skip = 1;  // Process every 2nd frame
    
    // Initialize camera with simpler pipeline
    cv::VideoCapture cap;
    
    // Try a few different ways to open the camera
    if (!cap.open(gstreamer_pipeline(camera_id, width, height), cv::CAP_GSTREAMER)) {
        std::cerr << "Failed to open with GStreamer pipeline, trying standard camera API" << std::endl;
        if (!cap.open(camera_id)) {
            std::cerr << "Failed to open camera with any method" << std::endl;
            return -1;
        }
    }
    
    // Load TorchScript model
    torch::jit::script::Module model;
    try {
        // Load the TorchScript model
        std::cout << "Loading TorchScript model..." << std::endl;
        model = torch::jit::load("/home/jetson/FaceRecognitionSystem/jetson/backend/python/saved_model/face_detection3_epoch200_loss0.1802_traced.ts");
        model.eval();
        
        // Move model to GPU if available
        torch::Device device(torch::kCUDA);
        if (torch::cuda::is_available()) {
            model.to(device);
            std::cout << "Using CUDA for model inference" << std::endl;
        } else {
            std::cout << "CUDA not available, using CPU" << std::endl;
            device = torch::Device(torch::kCPU);
        }
    } catch (const c10::Error& e) {
        std::cerr << "Error loading TorchScript model: " << e.what() << std::endl;
        return -1;
    }
    
    // Create window
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);
    
    // FPS calculation variables
    int frame_count = 0;
    double fps = 0.0;
    double freq = cv::getTickFrequency();
    double prev_tick = cv::getTickCount();
    
    // Frame counter for processing only some frames
    int frame_counter = 0;
    
    // Vector to store detected faces
    std::vector<cv::Rect> faces;
    
    // Device for inference
    torch::Device device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);
    
    while (true) {
        // Grab a frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            // Try to reconnect if frame grabbing fails
            std::cout << "Frame grab failed, attempting to reconnect..." << std::endl;
            cap.release();
            cv::waitKey(1000);  // Wait a bit before trying again
            if (!cap.open(gstreamer_pipeline(camera_id, width, height), cv::CAP_GSTREAMER)) {
                if (!cap.open(camera_id)) {
                    std::cerr << "Failed to reconnect to camera" << std::endl;
                    break;
                }
            }
            continue;
        }
        
        if (frame.empty()) {
            std::cerr << "Empty frame received" << std::endl;
            continue;
        }
        
        // Update frame counter and FPS calculation
        frame_count++;
        frame_counter++;
        
        // Only process every nth frame
        bool process_this_frame = (frame_counter % (frame_skip + 1) == 0);
        
        if (process_this_frame) {
            try {
                // Start timing
                double detect_start = cv::getTickCount();
                
                // Preprocess image for PyTorch model
                cv::Mat resized_frame;
                cv::resize(frame, resized_frame, cv::Size(320, 320)); // Adjust size to match model input
                
                // Convert from BGR to RGB
                cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2RGB);
                
                // Convert to tensor
                torch::Tensor input_tensor = torch::from_blob(
                    resized_frame.data, 
                    {1, resized_frame.rows, resized_frame.cols, 3}, // Shape: batch, height, width, channels
                    torch::kByte
                );
                
                // Transpose from NHWC to NCHW (what PyTorch expects)
                input_tensor = input_tensor.permute({0, 3, 1, 2});
                
                // Scale values to [0, 1]
                input_tensor = input_tensor.to(torch::kFloat32).div(255.0);
                
                // Move to device (GPU/CPU)
                input_tensor = input_tensor.to(device);
                
                // Run inference
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(input_tensor);
                
                torch::NoGradGuard no_grad; // Turn off gradients
                auto output = model.forward(inputs);
                
                // Process detections
                // This will depend on your model's output format
                // Here's a generic approach that should be adjusted based on your specific model
                
                // Clear previous faces
                faces.clear();
                
                // Option 1: If your model returns a dictionary
                if (output.isGenericDict()) {
                    auto output_dict = output.toGenericDict();
                    
                    // Extract boxes and scores
                    if (output_dict.contains("boxes") && output_dict.contains("scores")) {
                        auto boxes = output_dict.at("boxes").toTensor();
                        auto scores = output_dict.at("scores").toTensor();
                        
                        // Move tensors to CPU and convert to accessible format
                        boxes = boxes.to(torch::kCPU);
                        scores = scores.to(torch::kCPU);
                        
                        auto boxes_accessor = boxes.accessor<float, 2>();
                        auto scores_accessor = scores.accessor<float, 1>();
                        
                        // Process each detection
                        for (int i = 0; i < boxes_accessor.size(0); i++) {
                            if (scores_accessor[i] > 0.5) { // Confidence threshold
                                // Scale box to original frame
                                int x1 = static_cast<int>(boxes_accessor[i][0] * frame.cols / 320);
                                int y1 = static_cast<int>(boxes_accessor[i][1] * frame.rows / 320);
                                int x2 = static_cast<int>(boxes_accessor[i][2] * frame.cols / 320);
                                int y2 = static_cast<int>(boxes_accessor[i][3] * frame.rows / 320);
                                
                                faces.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                            }
                        }
                    }
                }
                // Option 2: If your model returns a tensor directly
                else if (output.isTensor()) {
                    auto detections = output.toTensor();
                    detections = detections.to(torch::kCPU);
                    
                    // Process based on your model's output format
                    // This is a generic example for SSD-style outputs:
                    // [batch_idx, class_id, confidence, x1, y1, x2, y2]
                    if (detections.dim() > 1) {
                        auto detections_accessor = detections.accessor<float, 3>(); 
                        
                        for (int i = 0; i < detections_accessor.size(1); i++) {
                            // Get confidence (index may vary based on your model output)
                            float confidence = detections_accessor[0][i][2];
                            
                            if (confidence > 0.5) { // Confidence threshold
                                // Get coordinates (indices may vary)
                                int x1 = static_cast<int>(detections_accessor[0][i][3] * frame.cols);
                                int y1 = static_cast<int>(detections_accessor[0][i][4] * frame.rows);
                                int x2 = static_cast<int>(detections_accessor[0][i][5] * frame.cols);
                                int y2 = static_cast<int>(detections_accessor[0][i][6] * frame.rows);
                                
                                faces.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                            }
                        }
                    }
                }
                // Option 3: If your model returns a list or tuple
                else if (output.isList()) {
                    auto output_list = output.toList();
                    
                    // Process based on your model's specific output format
                    // This depends heavily on what your model returns
                }
                
                // Calculate detection time
                double detect_time = (cv::getTickCount() - detect_start) / freq * 1000;
                
                // Show detection time
                std::stringstream detect_text;
                detect_text << "Detection: " << std::fixed << std::setprecision(1) << detect_time << " ms";
                cv::putText(frame, detect_text.str(), cv::Point(10, 60), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            } catch (const std::exception& e) {
                std::cerr << "Error during face detection: " << e.what() << std::endl;
            }
        }
        
        // Draw rectangles around detected faces
        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
        }
        
        // Calculate and display FPS
        double current_tick = cv::getTickCount();
        double time_diff = (current_tick - prev_tick) / freq;
        
        if (time_diff >= 1.0) {
            fps = frame_count / time_diff;
            frame_count = 0;
            prev_tick = current_tick;
        }
        
        std::stringstream fps_text;
        fps_text << "FPS: " << std::fixed << std::setprecision(1) << fps;
        cv::putText(frame, fps_text.str(), cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        
        // Show info about face detection
        std::stringstream faces_text;
        faces_text << "Faces: " << faces.size();
        cv::putText(frame, faces_text.str(), cv::Point(10, 90), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        // Display the resulting frame
        cv::imshow("Camera", frame);
        
        // Check for exit
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
    
    // Clean up
    cap.release();
    cv::destroyAllWindows();
    
    return 0;
}
