cmake_minimum_required(VERSION 3.25)
project(ColorVisionSimulation)

cmake_policy(SET CMP0076 NEW)
enable_testing()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(OpenCL REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_path(STB_INCLUDE_DIRS "stb_image.h")

add_subdirectory(lib)
add_subdirectory(bench)
add_subdirectory(test)

add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
