#include <model.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torchvision/vision.h>



void model::run_res10(int camera_id, int width, int height, int frame_skip) {

    // // Initialize camera with simpler pipeline
    cv::VideoCapture cap;

    if (!cap.open(camera::gstreamer_pipeline(camera_id, width, height), cv::CAP_GSTREAMER)) {
        std::cerr << "Failed to open with GStreamer pipeline, trying standard camera API" << std::endl;
        if (!cap.open(camera_id)) {
            std::cerr << "Failed to open camera with any method" << std::endl;
        }
    }
    
    // Load DNN face detector - more efficient than Haar cascades on Jetson
    // You'll need to download these model files first
    cv::dnn::Net faceNet;
    try {
        // Try to load the DNN model
        std::cout << "Loading DNN face detector..." << std::endl;
        faceNet = cv::dnn::readNetFromCaffe(
            "model/res10/deploy.prototxt",  
            "model/res10/res10_300x300_ssd_iter_140000.caffemodel");
        
        // If available, use CUDA but don't force it
        try {
            faceNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            faceNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout << "Using CUDA backend for DNN" << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "CUDA backend not available, using CPU: " << e.what() << std::endl;
            faceNet.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            faceNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading DNN model: " << e.what() << std::endl;
        std::cerr << "Please download the model files from: " << std::endl;
        std::cerr << "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector" << std::endl;
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
    
    while (true) {
        // Grab a frame
        cv::Mat frame;
        if (!cap.read(frame)) {
            // Try to reconnect if frame grabbing fails
            std::cout << "Frame grab failed, attempting to reconnect..." << std::endl;
            cap.release();
            cv::waitKey(1000);  // Wait a bit before trying again
            if (!cap.open(camera::gstreamer_pipeline(camera_id, width, height), cv::CAP_GSTREAMER)) {
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
        std::vector<cv::Rect> faces;
        
        if (process_this_frame) {
            try {
                // Start timing
                double detect_start = cv::getTickCount();
                
                // Prepare image for DNN
                cv::Mat inputBlob = cv::dnn::blobFromImage(
                    frame, 1.0, cv::Size(300, 300),
                    cv::Scalar(104.0, 177.0, 123.0), false, false);
                
                // Set input and forward pass
                faceNet.setInput(inputBlob);
                cv::Mat detection = faceNet.forward();
                
                // Process detections
                cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
                
                for (int i = 0; i < detectionMat.rows; i++) {
                    float confidence = detectionMat.at<float>(i, 2);
                    
                    // Filter weak detections
                    if (confidence > 0.5) {
                        int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                        int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                        int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                        int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
                        
                        // Add detected face
                        faces.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                    }
                }
                
                // Calculate detection time
                double detect_time = (cv::getTickCount() - detect_start) / freq * 1000;
                
                // Show detection time
                std::stringstream detect_text;
                detect_text << "Detection: " << std::fixed << std::setprecision(1) << detect_time << " ms";
                cv::putText(frame, detect_text.str(), cv::Point(10, 60), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            } catch (const cv::Exception& e) {
                std::cerr << "Error during face detection: " << e.what() << std::endl;
            }
        }
        
        // Draw rectangles around detected faces
        for (const auto& face : faces) {
            cv::Mat faceROI = frame(face);
            cv::imwrite("results/face.jpg", faceROI);
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
}

void model::run_kanface(int camera_id, int width, int height, int frame_skip) {

    std::string model_path = "/home/jetson/FaceRecognitionSystem/jetson/backend/model/kanface/kanface_06_25_512_trace_new.ts";
    torch::jit::script::Module model;
    torch::Device device(torch::kCPU);
    try {
        std::cout << "Loading TorchScript model..." << std::endl;
        if (torch::cuda::is_available()) {
            device = torch::Device(torch::kCUDA);
            std::cout << "Using CUDA for model inference" << std::endl;
        } else {
            std::cout << "Using CPU for model inference" << std::endl;
        }

        model = torch::jit::load(model_path);
        model.eval();
        model.to(device);

    } catch (const c10::Error& e) {
        std::cerr << "Error loading TorchScript model: " << e.what() << std::endl;
    }
}
