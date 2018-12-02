//
// Created by felix on 02.12.18.
//

#include "ReluLayer.h"

ReluLayer::ReluLayer() {
    setType(RELU_LAYER);
}

void ReluLayer::init() {
    inputHeight = getBeforeLayer()->getOutputHeight();
    inputWidth = getBeforeLayer()->getOutputWidth();
    inputDepth = getBeforeLayer()->getOutputDepth();

    outputHeight = inputHeight;
    outputWidth = inputWidth;
    outputDepth = inputDepth;
}

void ReluLayer::feedForward() {
    this->input = getBeforeLayer()->getOutput();

    output = arma::zeros(arma::size(input));
    //elementwise comparing of values -> either returning input value or zero from output cube
    output = arma::max(input, output);
}

void ReluLayer::backPropagate() {
    arma::cube upstreamGradient = getAfterLayer()->getGradientInput();

    gradientInput = input;
    gradientInput.transform([](double val) { return val > 0 ? 1 : 0; });
    gradientInput = gradientInput % upstreamGradient;
}

