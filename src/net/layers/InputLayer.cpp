#include "InputLayer.h"

InputLayer::InputLayer() {
    setType(INPUT_LAYER);
}

void InputLayer::init() {
    outputHeight = inputHeight;
    outputWidth = inputWidth;
    outputDepth = inputDepth;

    std::cout << "Output InputLayer: " << outputHeight << "x" << outputWidth << "x" << outputDepth << std::endl;

    output = arma::zeros(outputHeight, outputWidth, outputDepth);
}

void InputLayer::feedForward() {
    output = input;
}

void InputLayer::backPropagate() {
    //nothing to do here
}

