cmake_minimum_required(VERSION 3.10)
project(FaceRecognitionSystem)

set(CMAKE_PREFIX_PATH 
    "$ENV{CONDA_PREFIX}/lib/python3.6/site-packages/torch/share/cmake/Torch/" 
    # "$ENV{CONDA_PREFIX}/capstone/share/cmake/TorchVision/" 
)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_executable(main ${SRC_FILES} main.cpp)
# add_executable(main src/detection_with_caffee.cpp)

include_directories(include)

# Link OpenCV, Torch, and TIFF
target_compile_features(main PUBLIC cxx_range_for)
target_link_libraries(main PRIVATE "${OpenCV_LIBS}" "${TORCH_LIBRARIES}" TorchVision::TorchVision )
set_property(TARGET main PROPERTY CXX_STANDARD 14)
