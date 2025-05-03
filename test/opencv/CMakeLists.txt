cmake_minimum_required(VERSION 3.10)
project(test_opencv)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow -Werror -pedantic")
# file(GLOB_RECURSE SRC_FILES "../../src/.cpp")
find_package(OpenCV REQUIRED)

include_directories("../../include")

add_executable(test_opencv 
    "../../src/camera.cpp" 
    "../../src/cli.cpp" 
    test_opencv.cpp
)

target_link_libraries(test_opencv PRIVATE ${OpenCV_LIBS})

