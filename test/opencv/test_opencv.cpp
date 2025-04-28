#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <iostream>
#include <camera.h>
#include <cli.h>
#include <chrono>

using namespace cv;
using namespace cv::cuda;

int main(int argc, const char* argv[]) {
// int main() {
    // int device_count = cv::cuda::getCudaEnabledDeviceCount();
    // if (device_count == 0) {
    //     std::cout << "No CUDA-enabled devices found. OpenCV CUDA is NOT available.\n";
    // } else {
    //     std::cout << "CUDA-enabled devices: " << device_count << "\n";
    //     cv::cuda::DeviceInfo info;
    //     std::cout << "Using device: " << info.name() << "\n";
    // }
    cli::Args args{argc, argv};
    std::string model_path = args.get("--model");
    int camera_id = std::stoi(args.get("--camera_id", "0"));
    int width = std::stoi(args.get("--width", "1280"));
    int height = std::stoi(args.get("--height", "720"));
    int camera_fps = std::stoi(args.get("--fps", "60"));

    VideoCapture cap;
    if (!cap.open(camera::gstreamer_pipeline(camera_id, width, height, camera_fps), cv::CAP_GSTREAMER)) {
        std::cout << "cook" << std::endl;
    }

    Mat frame;
    int frame_counter = 0;
    double average_fps = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (true) {
        auto t1 = std::chrono::high_resolution_clock::now();
        cap >> frame;
        if (frame.empty()) {
            std::cout << "Error when capturing frame" << std::endl;
            break;
        }

        frame_counter++;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now -start_time).count();
        if (elapsed >= 1000) {
            average_fps = frame_counter * 1000.0 / elapsed;
            frame_counter = 0;
            start_time = now;
        }

        std::ostringstream stream;
        stream << std::fixed << std::setprecision(2) << average_fps;
        putText(frame, stream.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);


        imshow(":))", frame);
        if (waitKey(1) == 27) break;

        auto t2 = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "[Metric] Latency: " << latency << std::endl;
    }

    cap.release();
    destroyAllWindows();

    // Reading image
    // {
    // Mat image = imread("/home/jetson/FaceRecognitionSystem/jetson/backend/assets/0_Parade_marchingband_1_5.jpg");
    // if (image.empty()) {
    //     std::cout << "Can't read image" << std::endl;
    //     return -1;
    // }
    // imshow("1.jpg", image);
    // waitKey(0);
    // }
    // {
    // std::cout << getBuildInformation();
    // }
}


