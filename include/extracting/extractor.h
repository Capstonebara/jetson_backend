#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <torchvision/vision.h>
#include <opencv2/opencv.hpp>

class Extractor {
private:
    torch::jit::script::Module m_extractor;
    torch::Device m_device{torch::kCPU};
    static const int m_transforms_height = 112;
    static const int m_transforms_width = 112;
    static const std::vector<double> m_mean;
    static const std::vector<double> m_std;
public:
    Extractor(std::string &model_path);
    torch::Tensor inference(const torch::Tensor &input);
    static torch::Tensor preprocess(cv::Mat &roi);
};


#endif // EXTRACTOR_H
