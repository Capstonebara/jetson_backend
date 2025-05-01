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
#include <fstream>
#include <dirent.h>  // POSIX library for directory traversal
#include <sys/types.h> // For dirent.h (POSIX)
#include <cstring>    // For string manipulation
#include <map>        // For mapping users to embeddings


using namespace torch::indexing;
namespace utils {

// Structure to store user embedding info
struct UserEmbedding {
    int id;
    std::string name;
    int embedding_size;
    std::vector<float> embeddings;
};
void load_embeddings_from_bin(const std::string& file_path, utils::UserEmbedding& user_embedding);
void find_bin_files_recursive(const std::string& dir_path, std::vector<std::string>& bin_files);

std::vector<utils::UserEmbedding> load_all_embeddings(const std::string& base_directory);
torch::Tensor cvMatToTensor(cv::Mat &mat);
torch::Tensor transforms(cv::Mat &frame, cv::Size size, const std::vector<double> &mean, const std::vector<double> &std);
void printIValueType(const torch::IValue& val);
void printIValueRecursive(const torch::IValue& val, int indent = 0);
void printTensorInfo(torch::Tensor &tensor);

} // namespace utils
#endif // UTILS_H
