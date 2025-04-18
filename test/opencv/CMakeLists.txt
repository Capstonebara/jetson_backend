cmake_minimum_required(VERSION 3.10)
project(test_opencv)

file(GLOB_RECURSE SRC_FILES src/*.cpp)
find_package(OpenCV REQUIRED)

include_directories("../../include")

add_executable(test_opencv ${SRC_FILES} test_opencv.cpp)

target_link_libraries(test_opencv PRIVATE ${OpenCV_LIBS})
