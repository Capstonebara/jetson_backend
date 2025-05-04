
#include "logger.h"

Logger::Logger(const std::string& output_dir, VectorDB& db)
    : m_output_dir(output_dir), m_db(db) {}

void Logger::ensureDirectory(const std::string& path) {
    if (access(path.c_str(), F_OK) == -1) {
        mkdir(path.c_str(), 0775);  // single level
    }
}

std::string Logger::getCurrentTimestamp() {
    std::time_t t = std::time(nullptr);
    std::tm* now = std::localtime(&t);

    std::ostringstream oss;
    oss << std::setfill('0') 
        << now->tm_hour << ":" 
        << now->tm_mday << ":" 
        << (now->tm_mon + 1) << "." 
        << (now->tm_year + 1900);
    return oss.str();
}

void Logger::setDeviceID(int device_id) {
    m_device_id = device_id;
}

void Logger::logFaceAndTime(faiss::idx_t index, const cv::Mat& face_roi) {
    int user_id = m_db.getUserID(index);
    std::string user_dir = m_output_dir + "/";

    ensureDirectory(user_dir);

    std::string timestamp = getCurrentTimestamp();

    // Save image
    std::string image_path = user_dir + "saved_image.png";
    cv::imwrite(image_path, face_roi);

    // Save data.txt
    std::string data_txt_path = user_dir + "/data.txt";
    std::ofstream data_file(data_txt_path, std::ios::out | std::ios::trunc);
    if (data_file.is_open()) {
        data_file << user_id << " " 
            << m_device_id << " " 
            << timestamp << " " 
            << (m_isCheckIn ? "entry" : "exit") << std::endl;
        data_file.close();
    } else {
        std::cerr << "Logger Error: Failed to open " << data_txt_path << std::endl;
    }
}
