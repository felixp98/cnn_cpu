project(${CMAKE_PROJECT_NAME}_lib)

find_package(Armadillo 6.5 REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(HEADER_FILES
        net/Network.h utils/MnistDataLoader.h net/layers/Layer.h utils/Image.h net/activation/inc/ActivationFunction.h net/activation/inc/SigmoidActivation.h net/activation/inc/ReluActivation.h net/cost/inc/CostFunction.h net/cost/inc/QuadraticCost.h net/cost/inc/CrossEntropyCost.h  net/layers/ConvolutionalLayer.h net/layers/CrossEntropyLossLayer.h net/layers/FullyConnectedLayer.h net/layers/InputLayer.h net/layers/MaxPoolingLayer.h net/layers/ReluLayer.h net/layers/SigmoidLayer.h net/layers/SoftmaxLayer.h)

set(SOURCE_FILES
        net/Network.cpp utils/MnistDataLoader.cpp utils/Image.cpp net/activation/src/SigmoidActivation.cpp net/activation/src/ReluActivation.cpp net/cost/src/QuadraticCost.cpp net/cost/src/CrossEntropyCost.cpp net/layers/ConvolutionalLayer.cpp net/layers/CrossEntropyLossLayer.cpp net/layers/FullyConnectedLayer.cpp net/layers/InputLayer.cpp net/layers/MaxPoolingLayer.cpp net/layers/ReluLayer.cpp net/layers/SigmoidLayer.cpp net/layers/SoftmaxLayer.cpp)

#if (EXISTS ${SOURCE_FILES})
    # The library contains header and source files.
    add_library(${CMAKE_PROJECT_NAME}_lib STATIC
            ${SOURCE_FILES}
            ${HEADER_FILES}
            )
#else()
    # The library only contains header files.
#    add_library(${CMAKE_PROJECT_NAME}_lib INTERFACE)
#endif()
target_link_libraries(${CMAKE_PROJECT_NAME}_lib ${ARMADILLO_LIBRARIES})