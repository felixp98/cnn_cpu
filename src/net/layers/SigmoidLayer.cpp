#include "SigmoidLayer.h"

SigmoidLayer::SigmoidLayer() {
    setType(SIGMOID_LAYER);
}

void SigmoidLayer::init() {
    inputHeight = getBeforeLayer()->getOutputHeight();
    inputWidth = getBeforeLayer()->getOutputWidth();
    inputDepth = getBeforeLayer()->getOutputDepth();

    outputHeight = inputHeight;
    outputWidth = inputWidth;
    outputDepth = inputDepth;
}

void SigmoidLayer::feedForward() {
    this->input = getBeforeLayer()->getOutput();

    output = arma::zeros(arma::size(input));
    output = 1.0/(1.0 + arma::exp(-input));
}

void SigmoidLayer::backPropagate() {
    arma::cube upstreamGradient = getAfterLayer()->getGradientInput();

    gradientInput = arma::zeros(arma::size(input));
    gradientInput = output % (1.0-output);
    gradientInput = gradientInput % upstreamGradient;
}

