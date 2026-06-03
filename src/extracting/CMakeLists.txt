add_library(libextracting STATIC
    extractor.cpp
)

target_include_directories(libsearching PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
