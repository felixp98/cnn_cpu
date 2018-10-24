project(${CMAKE_PROJECT_NAME}_lib)

set(HEADER_FILES
        net/layers/inc/ConvolutionalLayer.h
        net/layers/inc/FullyConnectedLayer.h
        net/layers/inc/MaxPoolLayer.h
        net/layers/inc/ReluLayer.h
        net/layers/inc/SoftmaxLayer.h
    )

set(SOURCE_FILES
        net/layers/src/ConvolutionalLayer.cpp
        net/layers/src/FullyConnectedLayer.cpp
        net/layers/src/MaxPoolLayer.cpp
        net/layers/src/ReluLayer.cpp
        net/layers/src/SoftmaxLayer.cpp
    )

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