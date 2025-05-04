#include "preprocessor.h"
#include <iostream> 

Preprocessor::Preprocessor(float min_sharpness, int min_brightness, int max_brightness, float min_area_ratio, float max_area_ratio)
    : m_min_sharpness_threshold(min_sharpness),
      m_min_brightness_threshold(min_brightness),
      m_max_brightness_threshold(max_brightness),
      m_min_face_area_ratio(min_area_ratio),
      m_max_face_area_ratio(max_area_ratio)
      // target_size_ initialization removed
{
    // Optional: Add validation for input parameters
if (min_brightness < 0) {
    std::cerr << "Warning: min_brightness (" << min_brightness << ") is less than 0." << std::endl;
}
if (max_brightness > 255) {
    std::cerr << "Warning: max_brightness (" << max_brightness << ") is greater than 255." << std::endl;
}
if (min_brightness >= max_brightness) {
    std::cerr << "Warning: min_brightness (" << min_brightness << ") is greater than or equal to max_brightness (" << max_brightness << ")." << std::endl;
}
if ((1.0f / min_area_ratio) < 0.0f) {
    std::cerr << "Warning: 1 / min_area_ratio (" << min_area_ratio << ") is less than 0. Check min_area_ratio." << std::endl;
}
if ((1.0f / max_area_ratio) > 1.0f) {
    std::cerr << "Warning: 1 / max_area_ratio (" << max_area_ratio << ") is greater than 1.0. Check max_area_ratio." << std::endl;
}
if ( 1.0f / min_area_ratio >= 1.0f / max_area_ratio) {
    std::cerr << "Warning: min_area_ratio (" << min_area_ratio << ") is greater than or equal to max_area_ratio (" << max_area_ratio << ")." << std::endl;
}
}

// --- Private Helper Methods ---

float Preprocessor::computeSharpness(const cv::Mat& gray_roi) const {
    cv::Mat laplacian;
    cv::Laplacian(gray_roi, laplacian, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    return stddev.val[0] * stddev.val[0];
}

float Preprocessor::computeBrightness(const cv::Mat& gray_roi) const {
    cv::Scalar mean_brightness = cv::mean(gray_roi);
    return mean_brightness.val[0];
}

bool Preprocessor::isQualityAcceptable(const cv::Mat& gray_roi) const {
    float brightness = computeBrightness(gray_roi);
    if (brightness < m_min_brightness_threshold || brightness > m_max_brightness_threshold) {
        return false;
    }
    float sharpness = computeSharpness(gray_roi);
    if (sharpness < m_min_sharpness_threshold) {
        return false;
    }
    m_flag = true;
    return true;
}


// --- Public Processing Method ---

std::tuple<cv::Mat, cv::Rect, bool> Preprocessor::preprocessBestFace(const cv::Mat& frame, const torch::TensorAccessor<float, 2>& boxes_accessor, int transforms_width, int transforms_height)
{
    // --- Input Validation ---
    if (frame.empty() || frame.channels() != 3) {
        std::cerr << "Error: Input frame is empty or not 3-channel BGR." << std::endl;
        return std::make_tuple(cv::Mat(), cv::Rect(), false);
    }

    // --- Initialization ---
    cv::Mat best_face_roi_bgr; // Will store the final selected ROI
    cv::Rect box;
    float max_area = 0;
    long num_faces = boxes_accessor.size(0);

    if (num_faces == 0) {
        std::cout << "Their is no face" << std::endl;
        return std::make_tuple(cv::Mat(), cv::Rect(), false);
    } else {
        std::cout << "Frame has " << num_faces << " faces" << std::endl;
    }

    float total_image_area = static_cast<float>(frame.cols) * frame.rows;
    float min_allowed_area = total_image_area / m_min_face_area_ratio;
    float max_allowed_area = total_image_area / m_max_face_area_ratio;

    // --- Iterate through detected faces ---
    for (long i = 0; i < num_faces; ++i) {
        float scale_w = static_cast<float>(frame.cols) / transforms_width;
        float scale_h = static_cast<float>(frame.rows) / transforms_height;
        // 1. Extract and Validate Bounding Box Coordinates
        int x1 = static_cast<int>(std::round(boxes_accessor[i][0]) * scale_w);
        int y1 = static_cast<int>(std::round(boxes_accessor[i][1]) * scale_h);
        int x2 = static_cast<int>(std::round(boxes_accessor[i][2]) * scale_w);
        int y2 = static_cast<int>(std::round(boxes_accessor[i][3]) * scale_h);

        if (x2 <= x1 || y2 <= y1) {
            m_flag = false;
            continue;
        }

        // 2. Create cv::Rect and Clip to Image Bounds
        cv::Rect face_rect(x1, y1, x2 - x1, y2 - y1);
        cv::Rect clipped_rect = face_rect & cv::Rect(0, 0, frame.cols, frame.rows);

        if (clipped_rect.area() <= 0) {
            m_flag = false;
            continue;
        }
        

        // 3. Check Size Constraint
        float current_area = static_cast<float>(clipped_rect.area());
        if (current_area < min_allowed_area || current_area > max_allowed_area) {
            m_flag = false;
            continue;
        }

        // 4. Extract ROI and Convert to Grayscale
        cv::Mat current_roi_bgr = frame(clipped_rect);
        if (current_roi_bgr.empty()) {
            m_flag = false;
            continue;
        }
        cv::Mat current_roi_gray;
        cv::cvtColor(current_roi_bgr, current_roi_gray, cv::COLOR_BGR2GRAY);

        // 5. Check Quality (Sharpness and Brightness)
        if (!isQualityAcceptable(current_roi_gray)) {
            m_flag = false;
            continue;
        }

        // 6. Select if Largest Acceptable Face So Far
        if (current_area > max_area) {
            max_area = current_area;
            best_face_roi_bgr = current_roi_bgr.clone(); // Store the grayscale ROI (original size)
            box = clipped_rect;
        }
    } // End of loop through faces

    return std::make_tuple(best_face_roi_bgr, box, m_flag); // Return the processed ROI (or empty Mat if none selected)
}
