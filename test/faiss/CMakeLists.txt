cmake_minimum_required(VERSION 3.10)
project(test_faiss)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic")

set(CMAKE_PREFIX_PATH 
    "$ENV{CONDA_PREFIX}/lib/python3.6/site-packages/torch/share/cmake/Torch/" 
    # "/home/jetson/faiss/build/"
)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(OpenMP REQUIRED)
find_library(FAISS_LIB NAMES faiss PATHS /home/jetson/faiss/build/faiss/ )

include_directories("../../include" 
    # "/home/jetson/faiss/"
)
add_executable(test_faiss 
    "../../src/camera.cpp" 
    "../../src/cli.cpp" 
    test_faiss.cpp
)
target_link_libraries(test_faiss PRIVATE ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${FAISS_LIB} OpenMP::OpenMP_CXX openblas TorchVision::TorchVision)

set_property(TARGET test_faiss PROPERTY CXX_STANDARD 14)
