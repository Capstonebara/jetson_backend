#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaobjdetect.hpp>  // For CUDA-accelerated object detection
#include <opencv2/cudawarping.hpp>    // For CUDA-accelerated image processing
#include <opencv2/cudaimgproc.hpp>
#include <sstream>

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

std::string gstreamer_pipeline_usb(const std::string& device, int width, int height) {
    std::ostringstream pipeline;
    pipeline << "v4l2src device=" << device << " ! "
             << "image/jpeg, width=" << width << ", height=" << height << " ! "
             << "jpegdec ! videoconvert ! appsink";
    return pipeline.str();
}

int main(int argc, char* argv[]) {
    // Default values
    int camera_id = 0;
    int width = 1280;
    int height = 720;
    bool use_gstreamer = true;
    bool use_cuda = true;        // Flag to enable CUDA acceleration
    bool process_every_frame = false; // Process every frame or skip frames for speed
    int frame_skip = 2;          // Process 1 frame for every 3 frames
    int process_width = 640;     // Process at lower resolution for speed
    int process_height = 360;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--camera" && i + 1 < argc) {
            camera_id = std::stoi(argv[++i]);
        } else if (arg == "--width" && i + 1 < argc) {
            width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            height = std::stoi(argv[++i]);
        } else if (arg == "--use-usb") {
            use_gstreamer = false;
        } else if (arg == "--no-cuda") {
            use_cuda = false;
        } else if (arg == "--process-all") {
            process_every_frame = true;
        } else if (arg == "--skip" && i + 1 < argc) {
            frame_skip = std::stoi(argv[++i]);
        } else if (arg == "--process-width" && i + 1 < argc) {
            process_width = std::stoi(argv[++i]);
        } else if (arg == "--process-height" && i + 1 < argc) {
            process_height = std::stoi(argv[++i]);
        } else {
            std::cerr << "Usage: " << argv[0] << " [--camera <id>] [--width <value>] [--height <value>] [--use-usb] [--no-cuda] [--process-all] [--skip <n>] [--process-width <w>] [--process-height <h>]" << std::endl;
            return -1;
        }
    }

    // Initialize camera
    cv::VideoCapture cap;
    std::string device{"/dev/video" + std::to_string(camera_id)};
    if (use_gstreamer) {
        cap.open(gstreamer_pipeline(camera_id, width, height), cv::CAP_GSTREAMER);
    } else {
        cap.open(gstreamer_pipeline_usb(device, width, height), cv::CAP_GSTREAMER);
    }

    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the camera" << std::endl;
        return -1;
    }

    // Create a window to display the video
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);
    
    // Variables for FPS calculation
    int frame_count = 0;
    double fps = 0.0;
    double freq = cv::getTickFrequency();
    double prev_tick = cv::getTickCount();
    double fps_update_interval = 1.0; // Update FPS every 1 second

    // Load face detection model
    cv::CascadeClassifier face_cascade;
    cv::Ptr<cv::cuda::CascadeClassifier> cuda_face_cascade;
    
    if (use_cuda) {
        try {
            cuda_face_cascade = cv::cuda::CascadeClassifier::create("/home/jetson/opencv-4.5.2/data/haarcascades_cuda/haarcascade_profileface.xml");
            std::cout << "CUDA face cascade loaded successfully" << std::endl;
        } catch (const cv::Exception& e) {
            std::cerr << "Error loading CUDA face cascade: " << e.what() << std::endl;
            use_cuda = false;
        }
    }
    
    if (!use_cuda) {
        if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
            std::cerr << "Error loading face cascade" << std::endl;
            return -1;
        }
    }

    int frame_counter = 0;
    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Captured empty frame" << std::endl;
            break;
        }
        
        // Increment frame counter
        frame_count++;
        frame_counter++;
        
        // Process only every nth frame to improve speed
        bool process_this_frame = process_every_frame || (frame_counter % (frame_skip + 1) == 0);
        std::vector<cv::Rect> faces;
        
        if (process_this_frame) {
            double detect_start = cv::getTickCount();
            
            // Downscale for faster processing
            cv::Mat small_frame;
            cv::resize(frame, small_frame, cv::Size(process_width, process_height));
            
            // Convert to grayscale for faster processing
            cv::Mat gray;
            cv::cvtColor(small_frame, gray, cv::COLOR_BGR2GRAY);
            
            // Equalize the histogram to improve detection in varying lighting
            cv::equalizeHist(gray, gray);
            
            if (use_cuda) {
                // Use CUDA for detection
                cv::cuda::GpuMat gpu_gray(gray);
                cv::cuda::GpuMat gpu_faces;
                
                cuda_face_cascade->detectMultiScale(gpu_gray, gpu_faces);
                cuda_face_cascade->convert(gpu_faces, faces);
            } else {
                // CPU detection with optimized parameters
                face_cascade.detectMultiScale(
                    gray, faces, 
                    1.1,  // Scale factor - smaller for better detection but slower
                    3,    // Minimum neighbors - higher for fewer false positives
                    0,    // Flags (deprecated) 
                    cv::Size(30, 30)  // Minimum face size
                );
            }
            
            // Scale the face coordinates back to the original frame size
            for (auto& face : faces) {
                face.x = face.x * width / process_width;
                face.y = face.y * height / process_height;
                face.width = face.width * width / process_width;
                face.height = face.height * height / process_height;
            }
            
            double detect_end = cv::getTickCount();
            double detect_time = (detect_end - detect_start) / freq * 1000;
            
            // Draw detection time on the frame
            std::stringstream detect_text;
            detect_text << "Detection: " << std::fixed << std::setprecision(1) << detect_time << " ms";
            cv::putText(frame, detect_text.str(), cv::Point(10, 60), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        }
        
        // Draw rectangles around detected faces
        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
        }
        
        // Calculate FPS
        double current_tick = cv::getTickCount();
        double time_diff = (current_tick - prev_tick) / freq;
        
        if (time_diff >= fps_update_interval) {
            fps = frame_count / time_diff;
            frame_count = 0;
            prev_tick = current_tick;
        }
        
        // Display FPS and detection info
        std::stringstream fps_text;
        fps_text << "FPS: " << std::fixed << std::setprecision(1) << fps;
        cv::putText(frame, fps_text.str(), cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                    
        std::stringstream faces_text;
        faces_text << "Faces: " << faces.size();
        cv::putText(frame, faces_text.str(), cv::Point(10, 90), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
        
        if (!process_this_frame) {
            cv::putText(frame, "Skipping detection", cv::Point(10, 120), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 165, 255), 2);
        }

        cv::imshow("Camera", frame);

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
