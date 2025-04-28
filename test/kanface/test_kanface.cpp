#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <cli.h>
#include <utils.h>

using namespace torch::indexing;

int main(int argc, const char *argv[]) {
    cli::Args args{argc, argv};
    std::string model_path = args.get("--model", "/home/jetson/FaceRecognitionSystem/jetson/backend/model/kanface/kanface_06_25_512_trace_new_cuda.ts");
    std::string image_path = args.get("--image", "/home/jetson/FaceRecognitionSystem/jetson/backend/assets/Members.png");

    cv::Mat image = cv::imread(image_path);
    // for (int i = 0; i < 5; i++) {
    //     for (int j = 0; j < 5; j++) {
    //         std::cout << std::setw(4) << (int)image.at<cv::Vec3b>(i, j)[0] << " "; // Accessing Blue channel [0]
    //     }
    //     std::cout << std::endl;
    // }
    cv::Size size(112, 112);
    std::vector<double> mean{0.5, 0.5, 0.5};
    std::vector<double> std{0.5, 0.5, 0.5};

    torch::Tensor tensor = utils::transforms(image, size, mean, std); 

    // std::cout << "Tensor dtype: " << tensor.dtype() << "\n";
    // std::cout << "Tensor device: " << tensor.device() << "\n";
    // std::cout << "Tensor shape: " << tensor.sizes() << std::endl;
    // std::cout << tensor.is_contiguous() << std::endl;
    // std::cout << tensor.index({Slice(), 2, Slice(None, 5), Slice(None, 5)}) << std::endl;
    // std::cout << tensor.index({Slice(), 2, Slice(None, 5), Slice(None, 5)}).unsqueeze(0).sizes() << std::endl;

    torch::jit::script::Module model;
    try {
        model = torch::jit::load(model_path);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
    }

    model.eval();

    bool is_on_cuda = false;
    for (const auto& param : model.parameters()) {
        if (param.device().is_cuda()) {
            is_on_cuda = true;
            break;
        }
    }

    if (is_on_cuda && torch::cuda::is_available()) {
        torch::Device device{torch::kCUDA};
        tensor = tensor.to(device);
    } 

    std::vector<torch::IValue> inputs;
    inputs.push_back(tensor);

    torch::NoGradGuard no_grad;
    torch::Tensor output = model.forward(inputs).toTensor();

    std::cout << "Tensor device: " << output.device() << "\n";
    std::cout << "Tensor dtype: " << output.dtype() << "\n";
    std::cout << "Tensor shape: " << output.sizes() << std::endl;
    std::cout << output.is_contiguous() << std::endl;
    std::cout << output.index({Slice(), Slice(None, 10)}) << std::endl;

    return 0;
}

