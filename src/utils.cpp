#include <utils.h>

namespace utils {


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

void printIValueRecursive(const torch::IValue& val, int indent) {
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

void printTensorInfo(torch::Tensor &tensor) {
    std::cout << "tensor dtype: " << tensor.dtype() << "\n";
    std::cout << "tensor device: " << tensor.device() << "\n";
    std::cout << "tensor shape: " << tensor.sizes() << std::endl;
    std::cout << tensor.is_contiguous() << std::endl;
    // std::cout << tensor.index({Slice(), 2, Slice(None, 5), Slice(None, 5)}) << std::endl;
}
}
