#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <torch/torch.h> // Required for torch::TensorAccessor
#include <utility>

class Preprocessor {
public:
    // Constructor without target_size
    Preprocessor(float min_sharpness = 10.0,
                     int min_brightness = 50,
                     int max_brightness = 200,
                     float min_area_ratio = 0.004, // Corresponds roughly to 1/(15*15)
                     float max_area_ratio = 0.25); // Corresponds to 1/(2*2)

    // Main processing method
    std::tuple<cv::Mat, cv::Rect, bool> preprocessBestFace(const cv::Mat& frame,
                               const torch::TensorAccessor<float, 2>& boxes_accessor,
                               int transforms_width, 
                               int transforms_height
                               );

    mutable bool m_flag = false;
private:
    // Helper methods
    float computeSharpness(const cv::Mat& gray_roi) const;
    float computeBrightness(const cv::Mat& gray_roi) const;
    bool isQualityAcceptable(const cv::Mat& gray_roi) const;

    // Configuration Thresholds
    float m_min_sharpness_threshold;
    int m_min_brightness_threshold;
    int m_max_brightness_threshold;
    float m_min_face_area_ratio;
    float m_max_face_area_ratio;
};

#endif // PREPROCESSOR_H
