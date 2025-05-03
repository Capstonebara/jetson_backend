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
#include <detector.h>
#include <extractor.h>
#include <vectordb.h>
// #include <string>

namespace F = torch::nn::functional;
int main(int argc, const char *argv[]) { // int main() {
    cli::Args args{argc, argv};
    std::string detector_path = args.get("--detector_path", "/home/jetson/FaceRecognitionSystem/jetson/backend/model/ssdlite/ssdlite_320_cuda.ts");
    std::string extractor_path = args.get("--extractor_path", "/home/jetson/FaceRecognitionSystem/jetson/backend/model/kanface/kanface_06_25_128_custom_cuda.ts");
    // std::string image_path = args.get("--image", "/home/jetson/facerecognitionsystem/jetson/backend/assets/0_parade_marchingband_1_5.jpg");
    // std::string output_path = args.get("--output", "/home/jetson/facerecognitionsystem/jetson/backend/result/0_parade_marchingband_1_5_modified.jpg");
    const int camera_id = std::stoi(args.get("--camera-id", "0"));
    const int width = std::stoi(args.get("--width", "640"));
    const int height = std::stoi(args.get("--height", "480"));
    const int camera_fps = std::stoi(args.get("--fps", "60"));
    const int transforms_width = std::stoi(args.get("--transforms_width", "320"));
    const int transforms_height = std::stoi(args.get("--transforms_height", "320")); 
    const float detector_threshold = std::stof(args.get("--detector_threshold", "0.95"));
    const int skip_frame = std::stoi(args.get("--skip_frame", "2")); 
    const float extractor_threshold = std::stof(args.get("--extractor_threshold", "0.5"));
    const float progress_duration = std::stof(args.get("--progress_duration", "3000"));
    const std::string db_path = args.get("--db_path", "/home/jetson/FaceRecognitionSystem/jetson/backend/embeddings/");
    const double min_sharpness_threshold = std::stod(args.get("--min_sharpness_threshold", "50"));

    // ------ load database --------
    VectorDB db(db_path);

    // ------ load model --------

    Detector detector(detector_path);
    Extractor extractor(extractor_path);

    // ------ open camera --------
    cv::VideoCapture cap;
    if (!cap.open(camera::gstreamer_pipeline(camera_id, width, height, camera_fps), cv::CAP_GSTREAMER)) {
        std::cout << "failed to open camera" << std::endl;
    }

    cv::Mat frame;
    int frame_counter = 0;
    double average_fps = 0.0f;

    // std::vector<cv::Rect>    last_boxes;
    // std::vector<std::string> last_texts;

    std::unordered_map<int64_t, std::vector<torch::Tensor>> embedding_map;
    int64_t progress_time = 0;
    cv::Scalar bbox_color(0, 255, 255);

    std::string identity = "";

    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        auto t1 = std::chrono::high_resolution_clock::now();
        cap >> frame;
        ++frame_counter;

        if (frame.empty()) {
            std::cout << "failed to capture frame" << std::endl;
            break;
        }

        if (frame_counter % skip_frame == 0) {
            // last_boxes.clear();
            // last_texts.clear();

            auto progress_time_start = std::chrono::high_resolution_clock::now();
            const torch::Tensor &preprocessed_input = Detector::preprocess(frame, transforms_height, transforms_width);

            const torch::Tensor &selected_boxes = detector.inference(preprocessed_input, detector_threshold).to(torch::kCPU);

            auto accessor = selected_boxes.accessor<float, 2>();  // Use accessor on a CPU tensor

            cv::resize(frame, frame, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
            cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

            for (int64_t i = 0; i < accessor.size(0); ++i) {
                float scale_w = static_cast<float>(width) / transforms_width;
                float scale_h = static_cast<float>(height) / transforms_height;

                int x1 = static_cast<int>(std::round(accessor[i][0]) * scale_w);
                int y1 = static_cast<int>(std::round(accessor[i][1]) * scale_h);
                int x2 = static_cast<int>(std::round(accessor[i][2]) * scale_w);
                int y2 = static_cast<int>(std::round(accessor[i][3]) * scale_h);
                cv::Rect box(cv::Point(x1, y1), cv::Point(x2, y2));

                auto area = box.area();
                // std::cout << "area of frame: " << width * height << std::endl;
                // std::cout << "area of bbox: " << area << std::endl;
                // std::cout << "Frame larger than bbox " << width * height / area << " times" << std::endl;
                if ((area < (width * height / 15)) || (area > (width * height / 2))) { 
                    std::cout << "Embedding_map[" << i << "].size() = " << embedding_map[i].size() << std::endl;
                    embedding_map.erase(i);
                    progress_time = 0;
                    continue;
                }

                // last_boxes.push_back(box);

                // Extract embedding
                cv::Mat roi = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
                cv::Mat gray_roi;

                // --- Quality Check 2: Sharpness ---
                cv::cvtColor(roi, gray_roi, cv::COLOR_BGR2GRAY); // Assuming input `frame` is BGR for OpenCV display
                cv::Mat laplacian;
                cv::Laplacian(gray_roi, laplacian, CV_64F);
                cv::Scalar mu, sigma;
                cv::meanStdDev(laplacian, mu, sigma);
                double sharpness = sigma.val[0] * sigma.val[0];
                const double MIN_SHARPNESS_THRESHOLD = min_sharpness_threshold; // Tune this!
                if (sharpness < MIN_SHARPNESS_THRESHOLD) {
                    // Draw a box indicating ignored blur? (optional)
                    // cv::rectangle(frame, box, cv::Scalar(128, 128, 128), 1); // Gray for ignored blur
                    continue;
                }

                torch::Tensor preprocessed_roi = Extractor::preprocess(roi);
                torch::Tensor embedding = extractor.inference(preprocessed_roi);

                auto progress_time_end = std::chrono::high_resolution_clock::now();

                progress_time += std::chrono::duration_cast<std::chrono::milliseconds>(progress_time_end - progress_time_start ).count();
                // std::cout << "progress_time: " << progress_time << std::endl;

                if (progress_time < progress_duration) {
                    // draw background
                    cv::rectangle(frame, cv::Point(x1, y1 - int(height * 0.028)), cv::Point(x2, y1 - 5), cv::Scalar(202, 162, 177));
                    int progress_bar_width = static_cast<int>((progress_time / static_cast<float>(progress_duration)) * box.width);

                    // draw progress_bar 
                    cv::rectangle(frame, cv::Point(x1, y1 - int(height * 0.028)), cv::Point(x1 + progress_bar_width, y1 - 5), cv::Scalar(0, 255, 255), cv::FILLED);
                    cv::rectangle(frame, box, cv::Scalar(0, 255, 255), 2);
                    embedding_map[i].push_back(embedding);
                    continue;
                } 

                // calculate mean embedding
                std::vector<torch::Tensor>& embeddings = embedding_map[i];
                if (embeddings.size() == 0) {
                    continue;
                }
                torch::Tensor stacked = torch::stack(embeddings);
                torch::Tensor mean_embedding = stacked.mean(0);
                // std::cout << "Embedding vector: " << mean_embedding.index({Slice(), Slice(None, 10)})  << "\n";
                // // search bat dau tu day

                mean_embedding = mean_embedding.cpu().contiguous();  // always force contiguous memory
                mean_embedding = F::normalize(mean_embedding, F::NormalizeFuncOptions().p(2).dim(-1)); // always normalize because of ArcFace
                float* query_ptr = mean_embedding.data_ptr<float>();  // direct float* access

                // embedding = embedding.cpu().contiguous();  // always force contiguous memory
                // float* query_ptr = embedding.data_ptr<float>();  // direct float* access
                auto pair = db.search(query_ptr);
                faiss::idx_t index = pair.first;
                float similarity = pair.second;

                std::cout << "Found user: \"" << db.findName(index) 
                          << "\" at index: " << index
                          << " Similarity: " << similarity 
                          << std::endl;

                std::ostringstream ss;
                ss << db.findName(index)
                   << " " << std::fixed << std::setprecision(2)
                   << similarity;

                if (similarity > extractor_threshold) {
                    bbox_color = cv::Scalar(0, 255, 0);
                    cv::rectangle(frame, box, bbox_color, 2);
                    cv::putText(frame, ss.str(), cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 1, bbox_color, 1);
                } else {
                    bbox_color = cv::Scalar(0, 0, 255);
                    cv::rectangle(frame, box, bbox_color, 2);
                    cv::putText(frame, "Unknown", cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 1, bbox_color, 1);
                }
                // last_texts.push_back(ss.str());

            }
        // } else {
        //     for (size_t i = 0; i < last_boxes.size(); ++i) {
        //         auto area = last_boxes[i].area(); 
        //         if ((area > (width * height / 10)) && (area < (width * height / 2))) { 
        //         cv::rectangle(frame, last_boxes[i], bbox_color, 2); 
        //         cv::putText(frame, 
        //             last_texts[i],
        //             {last_boxes[i].x, last_boxes[i].y-5},
        //             cv::FONT_HERSHEY_SIMPLEX,
        //             1,
        //             bbox_color,
        //             2);
        //         } 
        //     }
        }

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (elapsed >= 1000) {
            average_fps = frame_counter * 1000.0 / elapsed;
            std::cout << "[Metric] Average FPS: " << average_fps << std::endl;
            frame_counter = 0;
            start_time = now;
        }

        std::ostringstream stream;
        stream << std::fixed << std::setprecision(2) << average_fps;
        cv::putText(frame, stream.str(), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("FaceRecognitionSystem", frame);

        auto t2 = std::chrono::high_resolution_clock::now();
        double latency = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        // std::cout << "[Metric] Latency: " << latency << std::endl;

        if (cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
