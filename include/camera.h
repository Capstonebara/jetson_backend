#ifndef CAMERA_H
#define CAMERA_H

#include <string>

namespace camera {

std::string gstreamer_pipeline(int sensor_id=0, int width=640, int height=480, int fps=60);

} // namespace camera

#endif // CAMERA_H
