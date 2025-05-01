#ifndef DETECTOR_H
#define DETECTOR_H

#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <torchvision/vision.h>
#include <opencv2/opencv.hpp>
#include <utils.h>

class Detector {
private:
    torch::jit::script::Module m_detector;
    torch::Device m_device{torch::kCPU};
    static const std::vector<double> m_mean;
    static const std::vector<double> m_std;

public:
    Detector(std::string &model_path);
    torch::Tensor inference(const torch::Tensor &input, const float threshold);
    static torch::Tensor preprocess(cv::Mat &roi, const int transforms_height, const int transforms_width);
};

#endif // DETECTOR_H
