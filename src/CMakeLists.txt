add_compile_options(-Wall -Wextra -Werror -pedantic -Wshadow -Wno-unused-parameter)

add_subdirectory(detecting)
add_subdirectory(extracting)
add_subdirectory(searching)

add_executable(main
    main.cpp
)

target_link_libraries(main PRIVATE
    libdetecting
    libextracting
    libsearching
    libutils
)

target_link_options(main PRIVATE "-Wl,--no-as-needed")
