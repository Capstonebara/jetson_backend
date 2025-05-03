cmake_minimum_required(VERSION 3.10)
project(test_opencv)

# Only set build type if not specified from outside (e.g., command line or CMake GUI)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Werror -pedantic")
set(CMAKE_PREFIX_PATH 
    "$ENV{CONDA_PREFIX}/lib/python3.6/site-packages/torch/share/cmake/Torch/" 
    # "$ENV{CONDA_PREFIX}/capstone/share/cmake/TorchVision/" 
)

# file(GLOB_RECURSE SRC_FILES "../../src/.cpp")
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(OpenMP REQUIRED)
find_library(FAISS_LIB NAMES faiss PATHS /home/jetson/faiss/build/faiss/ )

include_directories("../../include")
file(GLOB_RECURSE SRC_FILES "../../src/*.cpp")

add_executable(test_ssdlite
    ${SRC_FILES}
    test_ssdlite.cpp
)

target_link_libraries(test_ssdlite PRIVATE ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${FAISS_LIB} OpenMP::OpenMP_CXX openblas TorchVision::TorchVision)
set_property(TARGET test_ssdlite PROPERTY CXX_STANDARD 14)
