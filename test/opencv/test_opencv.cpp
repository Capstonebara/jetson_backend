#include <opencv2/opencv.hpp>
#include <iostream>
#include <camera.h>

using namespace cv;

int main() {
    Mat image = imread("/home/jetson/FaceRecognitionSystem/jetson/backend/assets/0_Parade_marchingband_1_5.jpg");
    if (image.empty()) {
        std::cout << "Can't read image" << std::endl;
        return -1;
    }
    imshow("1.jpg", image);
    waitKey(0);
    // std::cout << getBuildInformation();
}


