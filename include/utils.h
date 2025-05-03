#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <torchvision/vision.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>


using namespace torch::indexing;
namespace utils {

// Structure to store user embedding info

torch::Tensor cvMatToTensor(cv::Mat &mat);
torch::Tensor transforms(cv::Mat &frame, cv::Size size, const std::vector<double> &mean, const std::vector<double> &std);
void printIValueType(const torch::IValue& val);
void printIValueRecursive(const torch::IValue& val, int indent = 0);
void printTensorInfo(torch::Tensor &tensor);

} // namespace utils
#endif // UTILS_H
