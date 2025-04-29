#include <faiss/IndexFlat.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <opencv2/opencv.hpp>
#include <utils.h>
#include <cmath>
#include <chrono>
#include <unordered_map>
#include <camera.h>
#include <cli.h>
// #include <string>

using namespace torch::indexing;
int main(int argc, const char *argv[]) {
// int main() {
    cli::Args args{argc, argv};
    std::string detector_path = args.get("--detector_path", "/home/jetson/FaceRecognitionSystem/jetson/backend/model/ssdlite/ssdlite_320_cuda.ts");
    std::string extractor_path = args.get("--extractor_path", "/home/jetson/FaceRecognitionSystem/jetson/backend/model/kanface/kanface_06_25_128_custom_cuda.ts");
    // std::string image_path = args.get("--image", "/home/jetson/facerecognitionsystem/jetson/backend/assets/0_parade_marchingband_1_5.jpg");
    // std::string output_path = args.get("--output", "/home/jetson/facerecognitionsystem/jetson/backend/result/0_parade_marchingband_1_5_modified.jpg");
    int camera_id = std::stoi(args.get("--camera-id", "0"));
    int width = std::stoi(args.get("--width", "640"));
    int height = std::stoi(args.get("--height", "480"));
    int camera_fps = std::stoi(args.get("--fps", "60"));
    int transforms_width = std::stoi(args.get("--transforms_width", "320"));
    int transforms_height = std::stoi(args.get("--transforms_height", "320")); 
    float detector_threshold = std::stof(args.get("--detector_threshold", "0.95"));
    int skip_frame = std::stoi(args.get("--skip_frame", "2")); 
    float extractor_threshold = std::stof(args.get("--extractor_threshold", "1.2"));
    float progress_duration = std::stof(args.get("--progress_duration", "3"));
    std::string db_path = args.get("--db_path", "/home/jetson/FaceRecognitionSystem/jetson/backend/embeddings/");

    // ------ load database --------
    std::vector<utils::UserEmbedding> users = utils::load_all_embeddings(db_path);
    std::cout << "loaded embeddings for " << users.size() << " users" << std::endl;
    
    if (users.empty()) {
        std::cerr << "no user embeddings found. exiting." << std::endl;
        return 1;
    }

    int embedding_size = users.front().embedding_size;
    std::cout << "creating faiss index with dimension: " << embedding_size << std::endl;
    
    faiss::IndexFlatL2 index(embedding_size);
    
    // create a map to track which index entries belong to which user
    std::map<int, int> index_to_user_map;  // maps faiss index to user index in our vector
    
    // add all embeddings to the faiss index
    for (size_t user_idx = 0; user_idx < users.size(); ++user_idx) {
        const utils::UserEmbedding& user = users[user_idx];
        
        int num_vectors = user.embeddings.size() / embedding_size;
        std::cout << "adding " << num_vectors << " embeddings for user: " 
                  << user.name << " (id: " << user.id << ")" << std::endl;
        
        int start_idx = index.ntotal;
        index.add(num_vectors, user.embeddings.data());
        std::cout << "added vectors. index now contains " << index.ntotal << " vectors" << std::endl;
        
        // map the added indices to this user
        for (int i = 0; i < num_vectors; ++i) {
            index_to_user_map[start_idx + i] = user_idx;
        }
    }
    
    std::cout << "faiss index contains a total of " << index.ntotal << " vectors" << std::endl;
    


    // ------ open camera --------
    cv::VideoCapture cap;
    if (!cap.open(camera::gstreamer_pipeline(camera_id, width, height, camera_fps), cv::CAP_GSTREAMER)) {
        std::cout << "failed to open camera" << std::endl;
    }

    // ------ load model --------

    torch::jit::script::Module detector;
    try {
        detector = torch::jit::load(detector_path);

    } catch (const c10::Error &e) {
        std::cerr << "can't load model" << std::endl;
    }
    detector.eval();

    bool is_on_cuda = false;
    for (const auto& param : detector.parameters()) {
        if (param.device().is_cuda()) {
            is_on_cuda = true;
            break;
        }
    }
    

    torch::Device device_detector{torch::kCPU};
    if (is_on_cuda && torch::cuda::is_available()) {
        // dung non-blocking technique va  pin memory
        device_detector = torch::kCUDA;
    } 

    // // warm-up detector
    // {
    //     std::cout << "[Log] Start Warm-up detector " << std::endl;
    //     torch::NoGradGuard no_grad;
    //     torch::Tensor dummy_input = torch::rand({1, 3, 320, 320}).to(device_detector);
    //     detector.forward({dummy_input});
    //     std::cout << "[Log] End Warm-up detector " << std::endl;
    // }
    //
    //
    // // doan code nay duplicate can refactor
    torch::jit::script::Module extractor;
    try {
        extractor = torch::jit::load(extractor_path);

    } catch (const c10::Error &e) {
        std::cerr << "can't load model" << std::endl;
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

    // // warm-up extractor
    //
    // {
    //     std::cout << "[Log] Start Warm-up extractor " << std::endl;
    //     torch::NoGradGuard no_grad;
    //     torch::Tensor dummy_input = torch::rand({1, 3, 112, 112}).to(device_extractor);
    //     extractor.forward({dummy_input});
    //     std::cout << "[Log] End Warm-up extractor " << std::endl;
    // }


    // torch::tensor tensor{torch::rand({1, 3, 224, 224})};
    // torch::device device_detector{torch::kcuda};
    // std::cout << tensor.to(device_detector) << std::endl;
    // cv::Mat image = cv::imread(image_path);
    // cv::imshow("hello", image);

    // torch::tensor tensor = utils::transforms(image, size, mean, std); 

    cv::Size size_detector(transforms_width, transforms_height);
    std::vector<double> mean_detector{0.485, 0.456, 0.406};
    std::vector<double> std_detector{0.229, 0.224, 0.225};

    cv::Size size_extractor(112, 112);
    std::vector<double> mean_extractor{0.5, 0.5, 0.5};
    std::vector<double> std_extractor{0.5, 0.5, 0.5};

    // torch::tensor tensor = utils::transforms(image, size, mean_detector, std_detector); 
    cv::Mat frame;
    int frame_counter = 0;
    double average_fps = 0.0f;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<cv::Rect>    last_boxes;
    std::vector<std::string> last_texts;

    std::unordered_map<int64_t, std::vector<torch::Tensor>> embedding_map;
    int64_t progress_time = 0;

    // ██████╗ ███████╗████████╗███████╗ ██████╗████████╗ ██████╗ ██████╗ 
    // ██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
    // ██║  ██║█████╗     ██║   █████╗  ██║        ██║   ██║   ██║██████╔╝
    // ██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ██║   ██║██╔══██╗
    // ██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║
    // ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
    cv::Scalar bbox_color(0, 255, 255);

    while (true) {
        auto t1 = std::chrono::high_resolution_clock::now();
        cap >> frame;
        ++frame_counter;

        if (frame.empty()) {
            std::cout << "failed to capture frame" << std::endl;
            break;
        }
        if (frame_counter % skip_frame == 0) {
            last_boxes.clear();
            last_texts.clear();
            torch::Tensor tensor = utils::transforms(frame, size_detector, mean_detector, std_detector);

            // std::cout << "tensor dtype: " << tensor.dtype() << "\n";
            // std::cout << "tensor device: " << tensor.device() << "\n";
            // std::cout << "tensor shape: " << tensor.sizes() << std::endl;
            // std::cout << tensor.is_contiguous() << std::endl;
            // std::cout << tensor.index({slice(), 2, slice(None, 5), slice(None, 5)}) << std::endl;


            // create a vector of inputs
            std::vector<torch::IValue> inputs;
            inputs.push_back(tensor.to(device_detector));

            torch::NoGradGuard no_grad;

            // only declare `output` once
            auto output = detector.forward(inputs).toTuple();  // type: c10::intrusive_ptr<c10::ivalue::tuple>

            // utils::prinVivaluerecursive(detector.forward(inputs));
            // tensor.reset();
            // // only declare `detections` once
            std::vector<torch::IValue> detections = output->elements();
            //
            // // extract list of tensors
            auto bbox_list = detections[0].toList();   // list[tensor[4]]
            auto conf_list = detections[1].toList(); // list[tensor[]]
            // std::cout << bbox_list.get(0).toTensor().sizes() << std::endl;
            // std::cout << conf_list.get(0).toTensor().sizes() << std::endl;
            torch::Tensor boxes = bbox_list.get(0).toTensor();
            torch::Tensor mask = conf_list.get(0).toTensor().gt(detector_threshold);
            // std::cout << mask << std::endl;
            torch::Tensor selected_boxes = boxes.index({mask}).to(torch::kCPU);
            // std::cout << "size: "
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
                cv::Rect box(cv::Point(x1, y1), cv::Point(x2, y2));

                auto area = box.area();
                std::cout << "area of frame: " << width * height << std::endl;
                std::cout << "area of bbox: " << area << std::endl;
                std::cout << "area of frame / area of bbox: " << width * height / area << std::endl;
                if ( (area < (width * height / 8)) || (area > (width * height / 2))) {
                    embedding_map.erase(i);
                    continue;
                } 
                last_boxes.push_back(box);


                // std::cout << "tensor shape: " << tensor.sizes() << std::endl;
                // std::cout << "x1: " << x1 
                //     << " ,y1: " << y1
                //     << " ,x2: " << x2
                //     << " ,y2: " << y2
                //     << " tensor size(): " << tensor.sizes() 
                //     << std::endl;

                // ███████╗██╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗ ██████╗ ██████╗ 
                // ██╔════╝╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
                // █████╗   ╚███╔╝    ██║   ██████╔╝███████║██║        ██║   ██║   ██║██████╔╝
                // ██╔══╝   ██╔██╗    ██║   ██╔══██╗██╔══██║██║        ██║   ██║   ██║██╔══██╗
                // ███████╗██╔╝ ██╗   ██║   ██║  ██║██║  ██║╚██████╗   ██║   ╚██████╔╝██║  ██║

                auto progress_time_start = std::chrono::high_resolution_clock::now();
                cv::Mat roi = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                tensor = utils::transforms(roi, size_extractor, mean_extractor, std_extractor);

                inputs.push_back(tensor.to(device_extractor));
                //
                torch::Tensor embedding = extractor.forward(inputs).toTensor();
                

                auto progress_time_end = std::chrono::high_resolution_clock::now();

                progress_time += std::chrono::duration_cast<std::chrono::milliseconds>(progress_time_end - progress_time_start).count();
                std::cout << "progress_time: " << progress_time << std::endl;

                if (progress_time < progress_duration) {
                //     // draw background
                    cv::rectangle(frame, cv::Point(x1, y2 - 5), cv::Point(x2, y2 - 5), (50, 50, 50), cv::FILLED);
                    int progress_bar_width = progress_time / progress_duration * box.width;

                    // draw progress_bar 
                    cv::rectangle(frame, cv::Point(x1, y2 - 5), cv::Point(x1 + progress_bar_width, y2 - 5), (0, 255, 255), cv::FILLED);
                    cv::rectangle(frame, box, cv::Scalar(0, 255, 255), 2);
                //     embedding_map[i].push_back(embedding);
                    // continue;
                } 
                //
                // progress_time = 0;
                // if (embedding_map[i].empty()) {
                //     continue;
                // }

                // // calculate mean embedding
                // std::vector<torch::Tensor>& embeddings = embedding_map[i];
                // std::cout << "Embedding vector contain: " << embeddings.size() << "\n";
                // torch::Tensor stacked = torch::stack(embeddings);
                // torch::Tensor mean_embedding = stacked.mean(0);
                // std::cout << "Embedding vector: " << mean_embedding.index({Slice(), Slice(None, 10)})  << "\n";
                //
                // // search bat dau tu day
                // mean_embedding = mean_embedding.cpu().contiguous();  // always force contiguous memory
                // float* query_ptr = mean_embedding.data_ptr<float>();  // direct float* access

                embedding = embedding.cpu().contiguous();  // always force contiguous memory
                float* query_ptr = embedding.data_ptr<float>();  // direct float* access

                // std::cout << "tensor device: " << embedding.device() << "\n";
                // std::cout << "tensor dtype: " << embedding.dtype() << "\n";
                // std::cout << "tensor shape: " << embedding.sizes() << std::endl;
                std::cout << "query_ptr shape: " << sizeof(query_ptr) / sizeof(float) << std::endl;

                const int k = 1;  // number of nearest neighbors to find
                std::vector<float> distances(k);
                std::vector<faiss::idx_t> indices(k);

                index.search(1, query_ptr, k, distances.data(), indices.data());


                for (int i = 0; i < k; ++i) {
                    if (indices[i] >= 0 && indices[i] < index.ntotal) {
                        int user_idx = index_to_user_map[indices[i]];
                        std::cout << "match " << i << ": user " << users[user_idx].name 
                                  << " (dist: " << distances[i] << ")" << std::endl;
                        std::ostringstream ss;
                        ss << users[user_idx].name
                           << " " << std::fixed << std::setprecision(2)
                           << distances[i] * 1e5 << "e-05";
                        std::cout << "Distances: "  << distances[i] << std::endl;

                        if (distances[i] * 1e5 < extractor_threshold) {
                            bbox_color = cv::Scalar(0, 255, 0);
                            cv::rectangle(frame, box, bbox_color, 2);
                            cv::putText(frame, ss.str(), cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 1, bbox_color, 1);
                        } else {
                            bbox_color = cv::Scalar(0, 255, 255);
                            cv::rectangle(frame, box, bbox_color, 2);
                            cv::putText(frame, "Unknown", cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 1, bbox_color, 1);

                        }
                        last_texts.push_back(ss.str());

                    }
                }


                // std::cout << embedding.is_contiguous() << std::endl;
                // std::cout << embedding.index({Slice(), Slice(None, 10)}) << std::endl;
                // std::stringstream filename;
                // filename << "face_" << i << ".png";
                // cv::imwrite(filename.str(), roi);
            }
        } else {
            // ——— skipped frame: just re-draw last stored results ———

            for (size_t i = 0; i < last_boxes.size(); ++i) {
                auto area = last_boxes[i].area();
                if ((area > (width * height / 8)) && (area < (width * height / 2))) {
                    cv::rectangle(frame, last_boxes[i], bbox_color, 2);
                    cv::putText(frame,
                                last_texts[i],
                                {last_boxes[i].x, last_boxes[i].y-5},
                                cv::FONT_HERSHEY_SIMPLEX,
                                1,
                                bbox_color,
                                2);
                } 
            }
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
        cv::putText(frame, stream.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2);

        cv::imshow("facerecognitionsystem", frame);

        auto t2 = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "[metric] latency: " << latency << std::endl;

        if (cv::waitKey(17) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
