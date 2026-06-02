cmake_minimum_required(VERSION 3.18)
project(rtsp_trt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)   # RTX 4060; change to 86 for RTX 30xx

set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ── CUDA ──────────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── TensorRT ──────────────────────────────────────────────────────────────────
find_path(TRT_INCLUDE NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/local/tensorrt/include
    REQUIRED)
find_library(TRT_LIB nvinfer
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib
    REQUIRED)
find_library(TRT_PLUGIN nvinfer_plugin
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/tensorrt/lib)

# ── GStreamer ──────────────────────────────────────────────────────────────────
find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-video-1.0)

# ── OpenCV ────────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)

# ── yaml-cpp ──────────────────────────────────────────────────────────────────
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    find_library(YAML_CPP_LIB yaml-cpp REQUIRED)
    find_path(YAML_CPP_INC yaml-cpp/yaml.h REQUIRED)
    add_library(yaml-cpp UNKNOWN IMPORTED)
    set_target_properties(yaml-cpp PROPERTIES
        IMPORTED_LOCATION "${YAML_CPP_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${YAML_CPP_INC}")
endif()

# ── Target ────────────────────────────────────────────────────────────────────
add_executable(rtsp_trt
    src/main.cpp
    src/engine.cpp
    src/stream.cpp
    src/postprocess.cpp
    src/display.cpp
    src/preprocess.cu
    src/postprocess_gpu.cu
)

target_include_directories(rtsp_trt PRIVATE
    src/
    ${TRT_INCLUDE}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(rtsp_trt PRIVATE
    CUDA::cudart
    ${TRT_LIB}
    $<$<BOOL:${TRT_PLUGIN}>:${TRT_PLUGIN}>
    ${GST_LIBRARIES}
    ${OpenCV_LIBS}
    yaml-cpp
    pthread
)

target_compile_options(rtsp_trt PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
)
