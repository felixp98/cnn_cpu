cmake_minimum_required(VERSION 3.8)
project(cnn_gpu)

set(CMAKE_CXX_STANDARD 11)

#find_package(Armadillo 6.5 REQUIRED)
#include_directories(${ARMADILLO_INCLUDE_DIRS})

#set(SOURCE_FILES cnn_reference_test.cpp cnn_cpu_reference_LeNet.cpp)
add_executable(cnn_reference_test cnn_reference_test.cpp)

include_directories(src)

add_subdirectory(src)
add_subdirectory(tests)

target_link_libraries(cnn_reference_test ${CMAKE_PROJECT_NAME}_lib)

#ADD_SUBDIRECTORY(tests/lib/gtest-1.8)
#enable_testing()
#add_subdirectory(${gtest_SOURCE_DIR})
#include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
#include_directories(${gtest_SOURCE_DIR}/googlemock/include)

#set(SOURCE_FILES cnn_reference_test.cpp net/layers/src/ConvolutionalLayer.cpp tests/unit_tests/unit_tests.cpp)
#add_executable(cnn_gpu ${SOURCE_FILES})

#target_link_libraries(cnn_gpu gtest gtest_main)
#add_test(cnn_gpu cnn_gpu)
#target_link_libraries(cnn_gpu gmock gmock_main)