add_library(libsearching STATIC
    embedding_loader.cpp
    loader.cpp
    vectordb.cpp
)

target_include_directories(libsearching PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
