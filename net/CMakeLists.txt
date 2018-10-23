cmake_minimum_required(VERSION 3.8)
project(cnn_net)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        layers/src/ConvolutionalLayer.cpp
        layers/src/FullyConnectedLayer.cpp
        layers/src/MaxPoolLayer.cpp
        layers/src/ReluLayer.cpp
        layers/src/SoftmaxLayer.cpp)

target_link_libraries(cnn_gpu cnn_net)