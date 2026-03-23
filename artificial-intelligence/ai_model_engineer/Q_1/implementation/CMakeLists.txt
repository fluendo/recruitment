cmake_minimum_required(VERSION 3.16)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(Q_1_task LANGUAGES CXX C CUDA VERSION 1.0.0)

message("Compiler version: ${CMAKE_CXX_COMPILER_VERSION}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++2a -O3 \
        -Wno-deprecated-declarations \
        -Wno-maybe-uninitialized \
        -Wno-stringop-overflow \
        -Wno-unused-result")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} \
        -Wno-maybe-uninitialized \
        -Wno-stringop-overflow \
        -Wno-unused-result \
        -Wno-array-bounds")
endif()




message("Loading CUDA...")
find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_HOST_COMPILER gcc)

find_package(nlohmann_json REQUIRED)
find_package(fmt REQUIRED)

message("Loading OpenCV... ")
find_package(OpenCV 4 REQUIRED)

message("Loading TensorRT...")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
find_package(TensorRT REQUIRED)



include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
endif()

set(BUILD_SOURCES
    include/Types.hh
    include/Tensor.hh
    include/InferEngine.hh
    src/InferEngine.cc
    include/FaceDetectEngine.hh
    src/FaceDetectEngine.cc
    src/ReIdEngine.cc
    include/ReIdEngine.hh
    src/preproc.cu
    include/preproc.hh
    src/main.cc
)

add_library(
    ${PROJECT_NAME}
    SHARED
    ${BUILD_SOURCES}
)

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC include/
    PUBLIC ${CUDA_INCLUDE_DIRS}
    PUBLIC ${TensorRT_INCLUDE_DIRS}
    PUBLIC ${OpenCV_INCLUDE_DIRS}
)


target_link_libraries(
    ${PROJECT_NAME}
    dl
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    nvonnxparser
    ${CUDA_LIBRARIES}
    fmt::fmt
    nlohmann_json::nlohmann_json
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON LINKER_LANGUAGE CXX)

cuda_add_executable(test src/main.cc)
target_link_libraries(test ${PROJECT_NAME})