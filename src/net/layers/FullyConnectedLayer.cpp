//
// Created by felix on 02.12.18.
//

#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(size_t numOutputs)
: numOutputs(numOutputs) {
    setType(FULLY_CONNECTED_LAYER);
}

void FullyConnectedLayer::init() {
    //init in-/outputsize
    inputHeight = getBeforeLayer()->getOutputHeight();
    inputWidth = getBeforeLayer()->getOutputWidth();
    inputDepth = getBeforeLayer()->getOutputDepth();

    outputHeight = numOutputs;
    outputWidth = 1;
    outputDepth = 1;

    //std::cout << "Input FC: " << inputHeight << "x" << inputWidth << "x" << inputDepth << std::endl;
    std::cout << "Output FC: " << outputHeight << "x" << outputWidth << "x" << outputDepth << std::endl;

    output = arma::zeros(outputHeight, outputWidth, outputDepth);

    // Initialize the weights.
    weights = arma::zeros(numOutputs, inputHeight * inputWidth * inputDepth);
    //weights.imbue([&]() { return _getTruncNormalVal(0.0, 1.0); });
    weights.imbue([&]() { return getRandomValueBetweenBorders(-255, 255); });

    // Initialize the biases
    biases = arma::zeros(numOutputs);

    // Reset accumulated gradients.
    _resetAccumulatedGradients();
}

void FullyConnectedLayer::feedForward() {
    this->input = getBeforeLayer()->getOutput();

    arma::vec flatInput = arma::vectorise(input);
    output.slice(0).col(0) = (weights * flatInput) + biases;
    //output /= 100.0;
}

void FullyConnectedLayer::backPropagate() {
    arma::vec upstreamGradient = arma::vectorise(getAfterLayer()->getGradientInput());

    arma::vec gradInputVec = arma::zeros(inputHeight * inputWidth * inputDepth);
    for (size_t i = 0; i < (inputHeight * inputWidth * inputDepth); i++)
        gradInputVec[i] = arma::dot(weights.col(i), upstreamGradient);
    arma::cube tmp((inputHeight * inputWidth * inputDepth), 1, 1);
    tmp.slice(0).col(0) = gradInputVec;
    gradientInput = arma::reshape(tmp, inputHeight, inputWidth, inputDepth);

    gradWeights = arma::zeros(arma::size(weights));
    for (size_t i = 0; i < gradWeights.n_rows; i++)
        gradWeights.row(i) = vectorise(input).t() * upstreamGradient[i];

    accumulatedGradWeights += gradWeights;

    gradBiases = upstreamGradient;
    accumulatedGradBiases += gradBiases;
}

void FullyConnectedLayer::UpdateWeightsAndBiases(size_t batchSize, double learningRate) {
    weights = weights - learningRate * (accumulatedGradWeights / batchSize);
    biases = biases - learningRate * (accumulatedGradBiases / batchSize);
    _resetAccumulatedGradients();
}

double FullyConnectedLayer::_getTruncNormalVal(double mean, double variance) {
    double stddev = sqrt(variance);
    arma::mat candidate = {3.0 * stddev};
    while (std::abs(candidate[0] - mean) > 2.0 * stddev)
        candidate.randn(1, 1);
    return candidate[0];
}

void FullyConnectedLayer::_resetAccumulatedGradients() {
    accumulatedGradWeights = arma::zeros(numOutputs, inputHeight * inputWidth * inputDepth);
    accumulatedGradBiases = arma::zeros(numOutputs);
}

double FullyConnectedLayer::getRandomValueBetweenBorders(int min, int max){
    return ((rand()%(max-min + 1) + min)/(double)(max-min));
}