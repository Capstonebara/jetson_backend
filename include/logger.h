
#include <opencv2/opencv.hpp>
#include <string>
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <sys/stat.h>
#include <unistd.h>
#include "vectordb.h" 

class Logger {
public:
    Logger(const std::string& output_dir, VectorDB& db);

    void logFaceAndTime(faiss::idx_t index, const cv::Mat& face_roi);

    void setDeviceID(int device_id);

private:
    std::string m_output_dir;
    VectorDB& m_db;
    bool m_isCheckIn;
    int m_device_id;

    void ensureDirectory(const std::string& path);
    std::string getCurrentTimestamp();
};
