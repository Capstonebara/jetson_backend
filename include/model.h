#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <camera.h>
#include <cli.h>

namespace model {

// class Model {
//     private:
//         m_model_path;
//
//     public:
// };

void run_res10(int camera_id=0, int width=640, int height=480, int frame_skip=2);

void run_kanface(int camera_id=0, int width=640, int height=480, int frame_skip=2);


}


#endif // MODEL_H
