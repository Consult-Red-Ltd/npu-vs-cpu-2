cmake_minimum_required(VERSION 3.16)
project(benchmark_pipeline_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force C++17 onto subprojects (fixes Eigen errors inside TFLite)
set(CMAKE_CXX_EXTENSIONS OFF)
set(ABSL_PROPAGATE_CXX_STD ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    # QCS6490: Cortex-A78 (big) + Cortex-A55 (LITTLE)
    # +fp16+dotprod: FMLA fp16 and UDOT/SDOT int8 vector instructions
    # -flto=auto parallelises the LTRANS phase across all cores (plain -flto is serial)
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -mcpu=cortex-a78 -march=armv8.2-a+fp16+dotprod -ffast-math -flto=auto")
    string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " -flto=auto")
endif()

# ---- LiteRT (ai-edge-litert) ----
# The runtime moved from the TensorFlow source tree (tensorflow/lite) to the
# standalone LiteRT repo (github.com/google-ai-edge/litert). The TFLite core
# now lives under <litert>/tflite and builds the SAME CMake target name
# (tensorflow-lite), so linking it transitively exports the LiteRT repo root on
# the include path and #include "tflite/..." resolves. The C++ API itself is
# unchanged (namespace tflite::, TfLiteExternalDelegate* for the QNN delegate) -
# only the header prefix changed (tensorflow/lite/... -> tflite/...).
set(LITERT_SOURCE_DIR "$ENV{HOME}/litert-src" CACHE PATH "Path to the LiteRT source tree")

if(NOT LITERT_SOURCE_DIR OR NOT EXISTS "${LITERT_SOURCE_DIR}/tflite/CMakeLists.txt")
    message(FATAL_ERROR "LITERT_SOURCE_DIR must point to a valid LiteRT source tree "
                        "(git clone https://github.com/google-ai-edge/LiteRT.git).")
endif()

# TENSORFLOW_SOURCE_DIR is OPTIONAL: LiteRT's tflite CMake still needs a
# TensorFlow checkout for supporting headers (schema/TSL/XLA) and, if left
# empty, FetchContent-downloads the pinned tag (v2.21.0-rc0) itself. Pass it to
# reuse a local checkout and avoid re-downloading on every reconfigure.
set(TENSORFLOW_SOURCE_DIR "" CACHE PATH "Path to a TensorFlow source tree (optional; LiteRT fetches one if empty)")

# Avoid protobuf/tool target name collisions and keep the build lean.
set(TFLITE_ENABLE_PROFILING_PROTO OFF CACHE BOOL "Disable TFLite profiling proto target" FORCE)
set(TFLITE_ENABLE_BENCHMARK_MODEL OFF CACHE BOOL "Disable TFLite benchmark_model target" FORCE)
set(TFLITE_ENABLE_LABEL_IMAGE OFF CACHE BOOL "Disable TFLite label_image target" FORCE)

add_subdirectory(
    "${LITERT_SOURCE_DIR}/tflite"
    "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
    EXCLUDE_FROM_ALL
)

# ---- OpenCV ----
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio imgcodecs)

# ---- Main target ----
add_executable(benchmark_pipeline main.cpp)

target_include_directories(benchmark_pipeline PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(benchmark_pipeline PRIVATE
    ${OpenCV_LIBS}
    tensorflow-lite
    pthread
    dl
)

# OpenCV's videoio drags in FFmpeg's libavutil, which references versioned
# OpenCL symbols (clReleaseMemObject@OPENCL_1.0, ...). The board ships an
# Adreno OpenCL ICD (libOpenCL.so.1) that exports those symbols WITHOUT the
# @OPENCL_1.0 version nodes, so the static linker cannot satisfy them - even
# though we never exercise FFmpeg's OpenCL path (decode goes through
# GStreamer). Allow undefined symbols coming from shared libraries; they are a
# transitive dependency's concern and resolve at runtime.
target_link_options(benchmark_pipeline PRIVATE "-Wl,--allow-shlib-undefined")

install(TARGETS benchmark_pipeline RUNTIME DESTINATION bin)
