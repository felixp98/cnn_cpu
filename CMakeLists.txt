cmake_minimum_required(VERSION 3.8)
project(cnn_gpu)

set(CMAKE_CXX_STANDARD 11)

set(gtest_SOURCE_DIR lib/googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

set(SOURCE_FILES source/cnn_reference_test.cpp source/net/layers/src/ConvolutionalLayer.cpp tests/layer_tests.cpp)
add_executable(cnn_gpu ${SOURCE_FILES})

target_link_libraries(cnn_gpu gtest gtest_main)