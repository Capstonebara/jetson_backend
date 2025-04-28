cmake_minimum_required(VERSION 3.10)
project(test_faiss)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic")

set(CMAKE_PREFIX_PATH 
    "/home/jetson/miniforge-pypy3/envs/capstone/lib/python3.6/site-packages/torch/share/cmake/Torch/"
    # "/home/jetson/faiss/build/"
)
find_package(OpenMP REQUIRED)
find_library(FAISS_LIB NAMES faiss PATHS /home/jetson/faiss/build/faiss/ )

include_directories("../../include" 
    # "/home/jetson/faiss/"
)
add_executable(test_faiss 
    # "../../src/cli.cpp" 
    test_faiss.cpp
)
target_link_libraries(test_faiss PRIVATE ${FAISS_LIB} OpenMP::OpenMP_CXX openblas)

