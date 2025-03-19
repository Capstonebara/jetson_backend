#include <iostream>           // For standard input/output
#include <opencv2/opencv.hpp> // Include the OpenCV library
#include <opencv2/cudaobjdetect.hpp> // CUDA-based Haar Cascade
#include <sstream>

/* std::string gstreamer_pipeline(int sensor_id, int width, int height) { */
/*     std::ostringstream pipeline; */
/*     pipeline << "nvarguscamerasrc sensor-id=" << sensor_id << " ! " */
/*              << "video/x-raw(memory:NVMM), width=" << width << ", height=" << height << ", framerate=(fraction)60/1 ! " */
/*              << "nvvidconv flip-method=0 ! " */
/*              << "video/x-raw, width=" << width << ", height=" << height << ", format=(string)BGRx ! " */
/*              << "videoconvert ! " */
/*              << "video/x-raw, format=(string)BGR ! appsink "; */
/*     return pipeline.str(); */
/* } */

std::string gstreamer_pipeline(int sensor_id, int width, int height) {
    std::ostringstream pipeline;
    pipeline << "nvarguscamerasrc sensor-id=" << sensor_id << " ! "
             << "video/x-raw(memory:NVMM), width=" << width << ", height=" << height << ", framerate=(fraction)60/1 ! "
             << "nvvidconv flip-method=0 ! "
             << "video/x-raw(memory:NVMM), format=(string)NV12 ! "
             << "nvvidconv ! video/x-raw, format=(string)BGRx ! "
             << "videoconvert ! video/x-raw, format=(string)BGR ! "
             << "appsink sync=false drop=true max-buffers=2";
    /* 19/3/2025: max-buffers = 2 */ 
    /* This could lead to a situation where: */
    /* You detect faces in frame #1 */
    /* While you're trying to recognize those faces */
    /* Frame #1 gets replaced by frame #2, #3, or even later frames */
    /* The face coordinates you detected no longer match the current frame */

    /* By increasing to max-buffers=2, you allow: */

    /* One buffer for the frame you're currently processing */
    /* One buffer for the next incoming frame */

    /* This gives your system a small amount of breathing room to complete the full detection and recognition pipeline on a frame before it gets discarded, which can lead to more consistent results, especially when the recognition step takes significant processing time. */

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
    bool use_usb = false; 

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
            use_usb = true;
        } else {
            std::cerr << "Usage: " << argv[0] << " [--camera <id>] [--width <value>] [--height <value>] [--use-usb]" << std::endl;
            return -1;
        }
    }


    cv::VideoCapture cap;
    std::string device{"/dev/video" + std::to_string(camera_id)};
    if (use_usb) {
        cap.open(gstreamer_pipeline_usb(device, width, height), cv::CAP_GSTREAMER);
    } else {
        // Pass camera_id as sensor_id to the GStreamer pipeline
        cap.open(gstreamer_pipeline(camera_id, width, height), cv::CAP_GSTREAMER);
    }

    // Check if the camera is opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the camera" << std::endl;
        return -1;
    }

    // Create a window to display the video
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);  // Minimize buffer size
    cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);

    // Variables for FPS calculation
    int frame_count = 0;
    double fps = 0.0;
    double freq = cv::getTickFrequency();
    double prev_tick = cv::getTickCount();
    double fps_update_interval = 1.0; // Update FPS every 1 second

    while (true) {
        cv::Mat frame, gray; // Matrix to store the captured frame
        cap >> frame;  // Capture a new frame from the camera

        // Check if the frame is empty (end of stream or error)
        if (frame.empty()) {
            std::cerr << "Error: Captured empty frame" << std::endl;
            break;
        }

        // Increment frame counter
        frame_count++;
        
        // Calculate FPS
        double current_tick = cv::getTickCount();
        double time_diff = (current_tick - prev_tick) / freq;
        
        // Update FPS calculation every fps_update_interval seconds
        if (time_diff >= fps_update_interval) {
            fps = frame_count / time_diff;
            frame_count = 0;
            prev_tick = current_tick;
        }
        
        // Display FPS on frame
        std::stringstream fps_text;
        fps_text << "FPS: " << std::fixed << std::setprecision(2) << fps;
        cv::putText(frame, fps_text.str(), cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Camera", frame);

        // Wait for 1 ms and exit the loop if the user presses 'q'
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release the camera and destroy all OpenCV windows
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
