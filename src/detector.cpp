#include <detector.h>
const std::vector<double> Detector::m_mean{0.485, 0.456, 0.406};
const std::vector<double> Detector::m_std{0.229, 0.224, 0.225};

Detector::Detector(std::string &model_path) {
    try {
        m_detector = torch::jit::load(model_path);

    } catch (const c10::Error &e) {
        std::cerr << "Can't load detector" << std::endl;
    }

    m_detector.eval();

    bool is_on_cuda = false;
    for (const auto& param : m_detector.parameters()) {
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

torch::Tensor Detector::inference(const torch::Tensor &input, const float threshold) {

    std::vector<torch::IValue> inputs;
    inputs.push_back(input.to(m_device));

    torch::NoGradGuard no_grad;

    auto output = m_detector.forward(inputs).toTuple();
    std::vector<torch::IValue> detections = output->elements();
    auto bbox_list = detections[0].toList();
    auto conf_list = detections[1].toList();

    torch::Tensor boxes = bbox_list.get(0).toTensor();
    torch::Tensor mask = conf_list.get(0).toTensor().gt(threshold);

    torch::Tensor selected_boxes = boxes.index({mask});
    return selected_boxes;
}

torch::Tensor Detector::preprocess(cv::Mat &input, const int transforms_height, const int transforms_width) {
    return utils::transforms(input, cv::Size(transforms_height, transforms_width), m_mean, m_std);
}

// // warm-up detector
// {
//     std::cout << "[Log] Start Warm-up detector " << std::endl;
//     torch::NoGradGuard no_grad;
//     torch::Tensor dummy_input = torch::rand({1, 3, 320, 320}).to(device_detector);
//     detector.forward({dummy_input});
//     std::cout << "[Log] End Warm-up detector " << std::endl;
// }
