cmake_minimum_required(VERSION 3.10)
project(FaceRecognitionSystem LANGUAGES C CXX)

# ================================= #
# Compiler and build configuration  #
# ================================= #

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -Werror -pedantic")
# set(CMAKE_PREFIX_PATH 
#     "$ENV{CONDA_PREFIX}/lib/python3.6/site-packages/torch/share/cmake/Torch/" 
#     # "$ENV{CONDA_PREFIX}/capstone/share/cmake/TorchVision/" 
# )

# ======= #
# Testing #
# ======= #

option(BUILD_TESTING "Build unit tests" ON)

if(BUILD_TESTING)
    # Chỉ bật testing feature khi cần thiết
    enable_testing()
    message(STATUS "[Config] Building with Tests ENABLED")
else()
    message(STATUS "[Config] Building with Tests DISABLED")
endif()

# ============ #
# Dependencies #
# ============ #

# ====== #
# OpenCV #
# ====== #

find_package(OpenCV REQUIRED)

# ======== #
# LibTorch #
# ======== #

# find_package(Torch REQUIRED)
# find_package(TorchVision REQUIRED)

# file(GLOB_RECURSE SRC_FILES src/*.cpp)
include_directories(include)
add_executable(main ${SRC_FILES} src/main.cpp)
# add_executable(main src/detection_with_caffee.cpp)


# Link OpenCV, Torch, and TIFF
# target_compile_features(main PUBLIC cxx_range_for)
# target_link_libraries(main PRIVATE "${OpenCV_LIBS}" "${TORCH_LIBRARIES}" TorchVision::TorchVision)
# set_property(TARGET main PROPERTY CXX_STANDARD 14)

# ========================================== #
# Generate build configure for src and tests #
# ========================================== #

add_subdirectory(src)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# =================================== #
# Auto-Symlink compile_commands.json  #
# =================================== #

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(ORIGINAL_COMPILE_COMMANDS "${CMAKE_BINARY_DIR}/compile_commands.json")
  set(SYMLINK_COMPILE_COMMANDS
      "${CMAKE_SOURCE_DIR}/build/compile_commands.json")

  message(STATUS "[Tooling] Updating symlink for compile_commands.json...")

  file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/build")

  if(CMAKE_HOST_UNIX)
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E create_symlink "${ORIGINAL_COMPILE_COMMANDS}"
              "${SYMLINK_COMPILE_COMMANDS}")
  else()
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${ORIGINAL_COMPILE_COMMANDS}" "${SYMLINK_COMPILE_COMMANDS}")
  endif()

  message(
    STATUS
      "[Tooling] Symlinked: ${SYMLINK_COMPILE_COMMANDS} -> ${ORIGINAL_COMPILE_COMMANDS}"
  )
endif()
