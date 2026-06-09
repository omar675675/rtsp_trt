cmake_minimum_required(VERSION 3.18)
project(rtsp_trt LANGUAGES CXX CUDA)

# Locate pybind11 (installed via pip into the user site-packages)
execute_process(
    COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

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

# ── Shared pipeline sources (used by both the binary and the Python extension) ─
set(PIPELINE_SOURCES
    src/engine.cpp
    src/stream.cpp
    src/postprocess.cpp
    src/bytetrack.cpp
    src/pipeline_core.cpp
    src/preprocess.cu
    src/postprocess_gpu.cu
    src/crop.cu
)

set(PIPELINE_INCLUDES
    src/
    ${TRT_INCLUDE}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${GST_INCLUDE_DIRS}
)

set(PIPELINE_LIBS
    CUDA::cudart
    ${TRT_LIB}
    $<$<BOOL:${TRT_PLUGIN}>:${TRT_PLUGIN}>
    ${GST_LIBRARIES}
    yaml-cpp
    pthread
)

set(PIPELINE_COMPILE_OPTS
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wno-unused-parameter>
)

# ── CLI binary ────────────────────────────────────────────────────────────────
add_executable(rtsp_trt src/main.cpp ${PIPELINE_SOURCES})
target_include_directories(rtsp_trt PRIVATE ${PIPELINE_INCLUDES})
target_link_libraries(rtsp_trt PRIVATE ${PIPELINE_LIBS})
target_compile_options(rtsp_trt PRIVATE ${PIPELINE_COMPILE_OPTS})

# ── Python extension (.so) ────────────────────────────────────────────────────
pybind11_add_module(_rtsp_trt src/python_ext.cpp ${PIPELINE_SOURCES})
target_include_directories(_rtsp_trt PRIVATE ${PIPELINE_INCLUDES})
target_link_libraries(_rtsp_trt PRIVATE ${PIPELINE_LIBS})
target_compile_options(_rtsp_trt PRIVATE ${PIPELINE_COMPILE_OPTS})
# Place the .so at the project root so `import _rtsp_trt` works from there.
set_target_properties(_rtsp_trt PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)
