add_library(libdetecting STATIC
    detector.cpp
)

target_include_directories(libdetecting PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
