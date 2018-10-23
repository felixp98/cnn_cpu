include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

add_executable(runLayerTests layer_tests.cpp)

target_link_libraries(runLayerTests gtest gtest_main)
target_link_libraries(runLayerTests cnn_net)