#include <extractor.h>
#include <utils.h>

const std::vector<double> Extractor::m_mean{0.5, 0.5, 0.5};
const std::vector<double> Extractor::m_std{0.5, 0.5, 0.5};

Extractor::Extractor(std::string &model_path){
    try {
        m_extractor = torch::jit::load(model_path);

    } catch (const c10::Error &e) {
        std::cerr << "Can't load detector" << std::endl;
    }

    m_extractor.eval();

    bool is_on_cuda = false;
    for (const auto& param : m_extractor.parameters()) {
        if (param.device().is_cuda()) {
            is_on_cuda = true;
            break;
        }
    }
    

    if (is_on_cuda && torch::cuda::is_available()) {
        // dung non-blocking technique va  pin memory
        m_device = torch::kCUDA;
    } 

}

torch::Tensor Extractor::inference(const torch::Tensor &input) {

    std::vector<torch::IValue> inputs;
    inputs.push_back(input.to(m_device));

    torch::Tensor embedding = m_extractor.forward(inputs).toTensor();
    return embedding;
}

torch::Tensor Extractor::preprocess(cv::Mat &roi) {
    return utils::transforms(roi, cv::Size(m_transforms_height, m_transforms_width), m_mean, m_std);
}

// // warm-up extractor
//
// {
//     std::cout << "[Log] Start Warm-up extractor " << std::endl;
//     torch::NoGradGuard no_grad;
//     torch::Tensor dummy_input = torch::rand({1, 3, 112, 112}).to(device_extractor);
//     extractor.forward({dummy_input});
//     std::cout << "[Log] End Warm-up extractor " << std::endl;
// }
