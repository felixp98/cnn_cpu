#include "SoftmaxLayer.h"

SoftmaxLayer::SoftmaxLayer(size_t numInputs)
: numInputs(numInputs) {
    setType(SOFTMAX_LAYER);
}

void SoftmaxLayer::init() {
    //init in-/outputsize
    inputHeight = getBeforeLayer()->getOutputHeight();
    inputWidth = getBeforeLayer()->getOutputWidth();
    inputDepth = getBeforeLayer()->getOutputDepth();

    outputHeight = numInputs;
    outputWidth = 1;
    outputDepth = 1;

    //std::cout << "Input Softmax: " << inputHeight << "x" << inputWidth << "x" << inputDepth << std::endl;
    std::cout << "Output Softmax: " << outputHeight << "x" << outputWidth << "x" << outputDepth << std::endl;

    output = arma::zeros(outputHeight, outputWidth, outputDepth);
    gradientInput = arma::zeros(numInputs, 1, 1);
}

void SoftmaxLayer::feedForward() {
    this->input = getBeforeLayer()->getOutput();
    arma::vec flatInput = arma::vectorise(input);

    //flatInput.print();

    double sumExp = arma::accu(arma::exp(flatInput - arma::max(flatInput)));
    arma::vec temp = arma::exp(flatInput - arma::max(flatInput)) / sumExp;
    //temp.print();
    output.slice(0).col(0) = temp;
}

void SoftmaxLayer::backPropagate() {
    arma::vec upstreamGradient = arma::vectorise(getAfterLayer()->getGradientInput());
    arma::vec flatOutput = arma::vectorise(output);

    double sub = arma::dot(upstreamGradient, flatOutput);
    arma::vec temp = (upstreamGradient - sub) % flatOutput;
    gradientInput.slice(0).col(0) = temp;
}

