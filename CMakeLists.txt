cmake_minimum_required(VERSION 3.10)
project(FaceRecognitionSystem)

set(OpenCV_DIR ~/opencv-4.5.2/build)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

file(GLOB SRC_FILES src/*.cpp)

add_executable(main ${SRC_FILES})

target_link_libraries(main ${OpenCV_LIBS})

