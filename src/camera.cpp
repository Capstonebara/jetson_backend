#include <camera.h>
#include <sstream>

std::string camera::gstreamer_pipeline(int sensor_id, int width, int height) {
    std::ostringstream pipeline;
    pipeline << "nvarguscamerasrc sensor-id=" << sensor_id << " ! "
             << "video/x-raw(memory:NVMM), width=" << width << ", height=" << height << ", framerate=(fraction)60/1 ! "
             << "nvvidconv flip-method=0 ! "
             << "video/x-raw(memory:NVMM), format=(string)NV12 ! "
             << "nvvidconv ! video/x-raw, format=(string)BGRx ! "
             << "videoconvert ! video/x-raw, format=(string)BGR ! "
             << "appsink sync=false drop=true max-buffers=2";
    return pipeline.str();
}

