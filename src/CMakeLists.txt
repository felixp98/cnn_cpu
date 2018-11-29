project(${CMAKE_PROJECT_NAME}_lib)

find_package(Armadillo 6.5 REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(HEADER_FILES
        net/layers/inc/ConvolutionalLayer.h
        net/layers/inc/FullyConnectedLayer.h
        net/layers/inc/MaxPoolLayer.h
        net/layers/inc/SoftmaxLayer.h
        utils/inc/MnistDataLoader.h net/Network.h net/layers/inc/Layer.h utils/inc/Image.h net/activation/inc/ActivationFunction.h net/activation/inc/SigmoidActivation.h net/activation/inc/ReluActivation.h net/cost/inc/CostFunction.h net/cost/inc/QuadraticCost.h net/cost/inc/CrossEntropyCost.h)

set(SOURCE_FILES
        net/layers/src/ConvolutionalLayer.cpp
        net/layers/src/FullyConnectedLayer.cpp
        net/layers/src/MaxPoolLayer.cpp
        net/layers/src/SoftmaxLayer.cpp
        utils/src/MnistDataLoader.cpp net/Network.cpp utils/src/Image.cpp net/activation/src/SigmoidActivation.cpp net/activation/src/ReluActivation.cpp net/cost/src/QuadraticCost.cpp net/cost/src/CrossEntropyCost.cpp)

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