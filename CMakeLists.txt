cmake_minimum_required(VERSION 3.8)
project(cnn_gpu)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES src/cnn_reference_test.cpp)
add_executable(cnn_gpu ${SOURCE_FILES})