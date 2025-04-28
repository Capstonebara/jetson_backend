cmake_minimum_required(VERSION 3.10)
project(test_kanface)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Werror -pedantic")

set(CMAKE_PREFIX_PATH 
    "/home/jetson/miniforge-pypy3/envs/capstone/lib/python3.6/site-packages/torch/share/cmake/Torch/"
)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

include_directories("../../include")
add_executable(test_kanface 
    "../../src/cli.cpp" 
    test_kanface.cpp
)
target_link_libraries(test_kanface PRIVATE ${OpenCV_LIBS} ${Torch_Libraries} TorchVision::TorchVision)

