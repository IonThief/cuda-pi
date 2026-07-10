cmake_minimum_required(VERSION 3.18)

project(cuda-pi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "native")

set(SOURCES
    src/main.cu
    src/pi/core/base_method.cu
    src/pi/method/quadrature/riemann.cu
)
add_executable(cuda_pi ${SOURCES})

target_include_directories(cuda_pi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inc)
