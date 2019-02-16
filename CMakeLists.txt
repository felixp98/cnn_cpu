cmake_minimum_required(VERSION 3.8)
project(cnn_cpu)

set(CMAKE_CXX_STANDARD 11)

add_executable(cnn_reference_test cnn_reference_test.cpp)

include_directories(src)

add_subdirectory(src)
add_subdirectory(tests)

target_link_libraries(cnn_reference_test ${CMAKE_PROJECT_NAME}_lib)