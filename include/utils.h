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


namespace utils {

// Structure to store user embedding info
struct UserEmbedding {
    int id;
    std::string name;
    int embedding_size;
    std::vector<float> embeddings;
};

void load_embeddings_from_bin(const std::string& file_path, utils::UserEmbedding& user_embedding) {
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    int id;
    ifs.read(reinterpret_cast<char*>(&id), sizeof(int));
    user_embedding.id = id;

    int name_length;
    ifs.read(reinterpret_cast<char*>(&name_length), sizeof(int));
    std::vector<char> name_buf(name_length);
    ifs.read(name_buf.data(), name_length);
    user_embedding.name = std::string(name_buf.begin(), name_buf.end());

    int num_embeddings;
    int embedding_size;
    ifs.read(reinterpret_cast<char*>(&num_embeddings), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&embedding_size), sizeof(int));
    user_embedding.embedding_size = embedding_size;

    std::vector<float> embeddings(num_embeddings * embedding_size);
    ifs.read(reinterpret_cast<char*>(embeddings.data()),
             embeddings.size() * sizeof(float));
    user_embedding.embeddings = std::move(embeddings);
    
    ifs.close();
}

// Find all .bin files recursively in a directory
void find_bin_files_recursive(const std::string& dir_path, std::vector<std::string>& bin_files) {
    DIR* dir = opendir(dir_path.c_str());
    if (dir == nullptr) {
        std::cerr << "Failed to open directory: " << dir_path << std::endl;
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        
        // Skip . and .. directories
        if (name == "." || name == "..") {
            continue;
        }
        
        std::string full_path = dir_path + "/" + name;
        
        // Check if it's a directory
        if (entry->d_type == DT_DIR) {
            // Recursive call for subdirectories
            find_bin_files_recursive(full_path, bin_files);
        } 
        // Check if it's a .bin file
        else if (entry->d_type == DT_REG && name.size() > 4 && 
                 name.substr(name.size() - 4) == ".bin") {
            bin_files.push_back(full_path);
        }
    }
    
    closedir(dir);
}

std::vector<utils::UserEmbedding> load_all_embeddings(const std::string& base_directory) {
    std::vector<utils::UserEmbedding> all_user_embeddings;
    
    // Find all .bin files recursively
    std::vector<std::string> bin_files;
    find_bin_files_recursive(base_directory, bin_files);
    
    std::cout << "Found " << bin_files.size() << " .bin files" << std::endl;
    
    // Load each .bin file
    for (const auto& file_path : bin_files) {
        try {
            utils::UserEmbedding user_embedding;
            load_embeddings_from_bin(file_path, user_embedding);
            std::cout << "Loaded embedding from: " << file_path 
                      << " (ID: " << user_embedding.id 
                      << ", Name: " << user_embedding.name << ")" << std::endl;
            all_user_embeddings.push_back(std::move(user_embedding));
        } catch (const std::exception& e) {
            std::cerr << "Failed to load embedding file " << file_path << ": " << e.what() << std::endl;
        }
    }
    
    return all_user_embeddings;
}

cv::Mat gpuTensorToMat(torch::Tensor &tensor) {
    // Move to CPU and convert to uint8
    tensor = tensor.detach().to(torch::kCPU);
    
    // Assume input is float [0,1]; convert to uint8
    if (tensor.dtype() == torch::kFloat32) {
        tensor = tensor.mul(255).clamp(0, 255).to(torch::kUInt8);
    }

    // Make contiguous if needed
    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
    }
    // Shape should be (C, H, W) → convert to (H, W, C)
    TORCH_CHECK(tensor.dim() == 3, "Expected 3D tensor (C, H, W)");
    tensor = tensor.permute({1, 2, 0});  // (H, W, C)

    int height = tensor.size(0);
    int width = tensor.size(1);
    int channels = tensor.size(2);

    // Create OpenCV Mat header that shares memory with tensor
    // Be careful: this mat shares memory with the tensor
    cv::Mat mat(height, width, CV_MAKETYPE(CV_8U, channels), tensor.data_ptr());

    // Optional: return deep copy if you will destroy tensor soon
    return mat.clone();  // remove `.clone()` if you're careful with tensor lifetime
}

torch::Tensor cvMatToTensor(cv::Mat &mat)
{
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);

    // Step 2: Create tensor from mat.data on CPU
    auto tensor_cpu = torch::from_blob(
        mat.data,
        {1, mat.rows, mat.cols, mat.channels()},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU)
    ).permute({0, 3, 1, 2}).to(torch::kFloat32).contiguous().div_(255.0);  // [1, H, W, C] → [1, C, H, W]

    // mat.release();

    return tensor_cpu;
}

torch::Tensor transforms(cv::Mat &frame, 
                    cv::Size size, 
                    const std::vector<double> &mean, 
                    const std::vector<double> &std) {
    cv::resize(frame, frame, size, 0, 0, cv::INTER_LINEAR);
    torch::Tensor tensor = utils::cvMatToTensor(frame);
    tensor = torch::data::transforms::Normalize<>(mean, std)(tensor);
    return tensor;
}

void printIValueType(const torch::IValue& val) {
    if (val.isTensor()) std::cout << "Tensor" << std::endl;
    else if (val.isInt()) std::cout << "Int" << std::endl;
    else if (val.isDouble()) std::cout << "Double" << std::endl;
    else if (val.isBool()) std::cout << "Bool" << std::endl;
    else if (val.isTuple()) std::cout << "Tuple" << std::endl;
    else if (val.isList()) std::cout << "List" << std::endl;
    else if (val.isGenericDict()) std::cout << "Dict" << std::endl;
    else if (val.isString()) std::cout << "String" << std::endl;
    else std::cout << "Unknown Type" << std::endl;
}

void printIValueRecursive(const torch::IValue& val, int indent = 0) {
    auto pad = std::string(indent, ' ');
    if (val.isTensor()) {
        std::cout << pad << "Tensor of shape: " << val.toTensor().sizes() << "\n";
    } else if (val.isInt()) {
        std::cout << pad << "Int: " << val.toInt() << "\n";
    } else if (val.isDouble()) {
        std::cout << pad << "Double: " << val.toDouble() << "\n";
    } else if (val.isBool()) {
        std::cout << pad << "Bool: " << val.toBool() << "\n";
    } else if (val.isString()) {
        std::cout << pad << "String: " << val.toStringRef() << "\n";
    } else if (val.isList()) {
        std::cout << pad << "List:\n";
        auto list = val.toList();
        for (const auto& item : list) {
            printIValueRecursive(item, indent + 2);
        }
    } else if (val.isTuple()) {
        std::cout << pad << "Tuple:\n";
        auto tuple = val.toTuple();
        for (const auto& item : tuple->elements()) {
            printIValueRecursive(item, indent + 2);
        }
    } else if (val.isGenericDict()) {
        std::cout << pad << "Dict:\n";
        auto dict = val.toGenericDict();
        for (const auto& pair : dict) {
            std::cout << pad << " Key:\n";
            printIValueRecursive(pair.key(), indent + 2);
            std::cout << pad << " Value:\n";
            printIValueRecursive(pair.value(), indent + 2);
        }
    } else {
        std::cout << pad << "Unknown type\n";
    }
}


} // namespace utils
#endif // UTILS_H
