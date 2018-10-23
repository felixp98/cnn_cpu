cmake_minimum_required(VERSION 3.8)
project(cnn_gpu)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES cnn_reference_test.cpp)
add_executable(cnn_run ${SOURCE_FILES})

include_directories(cnn_net)

add_subdirectory(net)
add_subdirectory(tests)

target_link_libraries(cnn_run cnn_net)

#ADD_SUBDIRECTORY(tests/lib/gtest-1.8)
#enable_testing()
#add_subdirectory(${gtest_SOURCE_DIR})
#include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
#include_directories(${gtest_SOURCE_DIR}/googlemock/include)

#set(SOURCE_FILES cnn_reference_test.cpp net/layers/src/ConvolutionalLayer.cpp tests/layer_tests/layer_tests.cpp)
#add_executable(cnn_gpu ${SOURCE_FILES})

#target_link_libraries(cnn_gpu gtest gtest_main)
#add_test(cnn_gpu cnn_gpu)
#target_link_libraries(cnn_gpu gmock gmock_main)