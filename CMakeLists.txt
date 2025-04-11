cmake_minimum_required(VERSION 3.10)
project(FaceRecognitionSystem)

# set(CMAKE_PREFIX_PATH 
#     "/home/jetson/miniforge-pypy3/envs/capstone/lib/python3.6/site-packages/torch/share/cmake"
#     "/home/jetson/miniforge-pypy3/envs/capstone/share/cmake/TorchVision/"
# )
# set(OpenCV_DIR ~/opencv-4.5.2/build)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(TIFF REQUIRED)

# file(GLOB_RECURSE SRC_FILES main.cpp src/*.cpp)
# file(GLOB_RECURSE SRC_FILES main.cpp)
file(GLOB_RECURSE SRC_FILES src/utils/*.cpp test/test_load_detector.cpp)

add_executable(main ${SRC_FILES})

include_directories(include)

target_link_libraries(main "${OpenCV_LIBS}" "${TORCH_LIBRARIES}" TorchVision::TorchVision  TIFF::TIFF)

