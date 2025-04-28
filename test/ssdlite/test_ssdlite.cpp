#include <torch/script.h>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <opencv2/opencv.hpp>
#include <utils.h>
#include <cmath>
#include <chrono>
#include <camera.h>
#include <cli.h>
// #include <string>

using namespace torch::indexing;
int main(int argc, const char *argv[]) {
// int main() {
    cli::Args args{argc, argv};
    std::string detector_path = args.get("--detector_path", "/home/jetson/FaceRecognitionSystem/jetson/backend/model/ssdlite/ssdlite_320_cuda.ts");
    std::string extractor_path = args.get("--extractor_path", "/home/jetson/FaceRecognitionSystem/jetson/backend/model/kanface/kanface_06_25_512_trace_new_cuda.ts");
    // std::string image_path = args.get("--image", "/home/jetson/FaceRecognitionSystem/jetson/backend/assets/0_Parade_marchingband_1_5.jpg");
    // std::string output_path = args.get("--output", "/home/jetson/FaceRecognitionSystem/jetson/backend/result/0_Parade_marchingband_1_5_modified.jpg");
    int camera_id = std::stoi(args.get("--camera-id", "0"));
    int width = std::stoi(args.get("--width", "640"));
    int height = std::stoi(args.get("--height", "480"));
    int camera_fps = std::stoi(args.get("--fps", "60"));
    int transforms_width = std::stoi(args.get("--transforms_width", "320"));
    int transforms_height = std::stoi(args.get("--transforms_height", "320")); 
    float threshold = std::stof(args.get("--threshold", "0.8"));

    // ------ Open camera --------
    cv::VideoCapture cap;
    if (!cap.open(camera::gstreamer_pipeline(camera_id, width, height, camera_fps), cv::CAP_GSTREAMER)) {
        std::cout << "Failed to open camera" << std::endl;
    }

    // ------ Load model --------

    torch::jit::script::Module detector;
    try {
        detector = torch::jit::load(detector_path);

    } catch (const c10::Error &e) {
        std::cerr << "Can't load model" << std::endl;
    }
    detector.eval();

    bool is_on_cuda = false;
    for (const auto& param : detector.parameters()) {
        if (param.device().is_cuda()) {
            is_on_cuda = true;
            break;
        }
    }
    
    // Doan code nay duplicate can refactor
    torch::Device device_detector{torch::kCPU};
    if (is_on_cuda && torch::cuda::is_available()) {
        // dung non-blocking technique va  pin memory
        device_detector = torch::kCUDA;
    } 


    torch::jit::script::Module extractor;
    try {
        extractor = torch::jit::load(extractor_path);

    } catch (const c10::Error &e) {
        std::cerr << "Can't load model" << std::endl;
    }
    extractor.eval();

    bool is_on_cuda_extractor = false;
    for (const auto& param : extractor.parameters()) {
        if (param.device().is_cuda()) {
            is_on_cuda_extractor = true;
            break;
        }
    }

    torch::Device device_extractor{torch::kCPU};
    if (is_on_cuda_extractor && torch::cuda::is_available()) {
        // dung non-blocking technique va  pin memory
        device_extractor = torch::kCUDA;
    } 
    // torch::Tensor tensor{torch::rand({1, 3, 224, 224})};
    // torch::Device device_detector{torch::kCUDA};
    // std::cout << tensor.to(device_detector) << std::endl;
    // cv::Mat image = cv::imread(image_path);
    // cv::imshow("hello", image);
    
    // torch::Tensor tensor = utils::transforms(image, size, mean, std); 

    cv::Size size_detector(transforms_width, transforms_height);
    std::vector<double> mean_detector{0.485, 0.456, 0.406};
    std::vector<double> std_detector{0.229, 0.224, 0.225};

    cv::Size size_extractor(112, 112);
    std::vector<double> mean_extractor{0.5, 0.5, 0.5};
    std::vector<double> std_extractor{0.5, 0.5, 0.5};

    // torch::Tensor tensor = utils::transforms(image, size, mean_detector, std_detector); 
    cv::Mat frame;
    int frame_counter = 0;
    double average_fps = 0.0F;
    auto start_time = std::chrono::high_resolution_clock::now();

    // ██████╗ ███████╗████████╗███████╗ ██████╗████████╗ ██████╗ ██████╗ 
    // ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
    // ██║  ██║█████╗     ██║   █████╗  ██║        ██║   ██║   ██║██████╔╝
    // ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ██║   ██║██╔══██╗
    // ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║
    // ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
    while (true) {
        auto t1 = std::chrono::high_resolution_clock::now();
        cap >> frame;
        ++frame_counter;

        if (frame.empty()) {
            std::cout << "Failed to capture frame" << std::endl;
            break;
        }
        torch::Tensor tensor = utils::transforms(frame, size_detector, mean_detector, std_detector);

        // std::cout << "Tensor dtype: " << tensor.dtype() << "\n";
        // std::cout << "Tensor device: " << tensor.device() << "\n";
        // std::cout << "Tensor shape: " << tensor.sizes() << std::endl;
        // std::cout << tensor.is_contiguous() << std::endl;
        // std::cout << tensor.index({Slice(), 2, Slice(None, 5), Slice(None, 5)}) << std::endl;


        // Create a vector of inputs
        std::vector<torch::IValue> inputs;
        inputs.push_back(tensor.to(device_detector));

        torch::NoGradGuard no_grad;

        // Only declare `output` once
        auto output = detector.forward(inputs).toTuple();  // type: c10::intrusive_ptr<c10::ivalue::Tuple>

        // utils::printIValueRecursive(detector.forward(inputs));
        // tensor.reset();
        // // Only declare `detections` once
        std::vector<torch::IValue> detections = output->elements();
        //
        // // Extract list of tensors
        auto bbox_list = detections[0].toList();   // List[Tensor[4]]
        auto conf_list = detections[1].toList(); // List[Tensor[]]
        // std::cout << bbox_list.get(0).toTensor().sizes() << std::endl;
        // std::cout << conf_list.get(0).toTensor().sizes() << std::endl;
        torch::Tensor boxes = bbox_list.get(0).toTensor();
        torch::Tensor mask = conf_list.get(0).toTensor().gt(threshold);
        // std::cout << mask << std::endl;
        torch::Tensor selected_boxes = boxes.index({mask}).to(torch::kCPU);
        // std::cout << "Size: "
        //     << selected_boxes.sizes() 
        //     << "\n"
        //     << selected_boxes 
        //     << std::endl;

        auto accessor = selected_boxes.accessor<float, 2>();
        cv::resize(frame, frame, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
        for (int64_t i = 0; i < accessor.size(0); ++i) {
            inputs.clear();
            float scale_w = static_cast<float>(width) / size_detector.width;
            float scale_h = static_cast<float>(height) / size_detector.height;

            int x1 = static_cast<int>(std::round(accessor[i][0]) * scale_w);
            int y1 = static_cast<int>(std::round(accessor[i][1]) * scale_h);
            int x2 = static_cast<int>(std::round(accessor[i][2]) * scale_w);
            int y2 = static_cast<int>(std::round(accessor[i][3]) * scale_h);
            int area = (y2 - y1) * (x2 - x1);
            std::cout << "width * height: " << width * height << std::endl;
            std::cout << "area: " << area << std::endl;
            std::cout << "percentage: " << width * height / area << std::endl;
            if ( (area > (width * height / 8)) && (area < (width * height / 2))) {
                cv::Rect box(cv::Point(x1, y1), cv::Point(x2, y2));
                cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
            } else {
                continue;
            }

            
            // std::cout << "Tensor shape: " << tensor.sizes() << std::endl;
            // std::cout << "x1: " << x1 
            //     << " ,y1: " << y1
            //     << " ,x2: " << x2
            //     << " ,y2: " << y2
            //     << " Tensor size(): " << tensor.sizes() 
            //     << std::endl;
            //
            // // ███████╗██╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗ ██████╗ ██████╗ 
            // // ██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
            // // █████╗   ╚███╔╝    ██║   ██████╔╝███████║██║        ██║   ██║   ██║██████╔╝
            // // ██╔══╝   ██╔██╗    ██║   ██╔══██╗██╔══██║██║        ██║   ██║   ██║██╔══██╗
            // // ███████╗██╔╝ ██╗   ██║   ██║  ██║██║  ██║╚██████╗   ██║   ╚██████╔╝██║  ██║
            //
            // cv::Mat roi = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
            // tensor = utils::transforms(roi, size_extractor, mean_extractor, std_extractor);
            //
            // inputs.push_back(tensor.to(device_extractor));
            //
            // torch::Tensor embedding = extractor.forward(inputs).toTensor();
            //
            // std::cout << "Tensor device: " << embedding.device() << "\n";
            // std::cout << "Tensor dtype: " << embedding.dtype() << "\n";
            // std::cout << "Tensor shape: " << embedding.sizes() << std::endl;
            // std::cout << embedding.is_contiguous() << std::endl;
            // std::cout << embedding.index({Slice(), Slice(None, 10)}) << std::endl;
            // std::stringstream filename;
            // filename << "face_" << i << ".png";
            // cv::imwrite(filename.str(), roi);
        }


        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (elapsed >= 1000) {
            average_fps = frame_counter * 1000.0 / elapsed;
            frame_counter = 0;
            start_time = now;
        }

        std::ostringstream stream;
        stream << std::fixed << std::setprecision(2) << average_fps;
        cv::putText(frame, stream.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("FaceRecognitionSystem", frame);

        auto t2 = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "[Metric] Latency: " << latency << std::endl;

        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
