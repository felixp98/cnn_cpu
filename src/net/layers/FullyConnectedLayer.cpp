#include "FullyConnectedLayer.h"
#include <chrono>

#define TIME_MEASURE true

#if TIME_MEASURE
using namespace std::chrono;
#endif

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
    //weights.imbue([&]() { return getRandomWeight(0.0, 1.0); });
    weights.imbue([&]() { return getRandomValueBetweenBorders(-255, 255); });

    // Initialize the biases
    biases = arma::zeros(numOutputs);

    // Reset accumulated gradients.
    resetAccumulatedNablas();
}

void FullyConnectedLayer::feedForward() {
#if TIME_MEASURE
    high_resolution_clock::time_point t_forward_start = high_resolution_clock::now();
#endif

    this->input = getBeforeLayer()->getOutput();

    arma::vec flatInput = arma::vectorise(input);
    output.slice(0).col(0) = (weights * flatInput) + biases;
    //output /= 100.0;

#if TIME_MEASURE
    high_resolution_clock::time_point t_forward_stop = high_resolution_clock::now();
    forwardDuration += duration_cast<microseconds>(t_forward_stop-t_forward_start).count();
#endif
}

void FullyConnectedLayer::backPropagate() {
#if TIME_MEASURE
    high_resolution_clock::time_point t_backward_start = high_resolution_clock::now();
#endif

    arma::vec upstreamGradient = arma::vectorise(getAfterLayer()->getGradientInput());

    arma::vec gradInputVec = arma::zeros(inputHeight * inputWidth * inputDepth);
    for (size_t i = 0; i < (inputHeight * inputWidth * inputDepth); i++)
        gradInputVec[i] = arma::dot(weights.col(i), upstreamGradient);
    arma::cube tmp((inputHeight * inputWidth * inputDepth), 1, 1);
    tmp.slice(0).col(0) = gradInputVec;
    gradientInput = arma::reshape(tmp, inputHeight, inputWidth, inputDepth);

    nablaWeights = arma::zeros(arma::size(weights));
    for (size_t i = 0; i < nablaWeights.n_rows; i++)
        nablaWeights.row(i) = vectorise(input).t() * upstreamGradient[i];

    accumulatedNablaWeights += nablaWeights;

    nablaBiases = upstreamGradient;
    accumulatedNablaBiases += nablaBiases;

#if TIME_MEASURE
    high_resolution_clock::time_point t_backward_stop = high_resolution_clock::now();
    backwardDuration += duration_cast<microseconds>(t_backward_stop-t_backward_start).count();
#endif
}

void FullyConnectedLayer::updateWeightsAndBiases(size_t batchSize, double learningRate) {
    weights = weights - learningRate * (accumulatedNablaWeights / batchSize);
    biases = biases - learningRate * (accumulatedNablaBiases / batchSize);
    resetAccumulatedNablas();
}

double FullyConnectedLayer::getRandomWeight(double mean, double variance) {
    double stddev = sqrt(variance);
    arma::mat candidate = {3.0 * stddev};
    while (std::abs(candidate[0] - mean) > 2.0 * stddev)
        candidate.randn(1, 1);
    return candidate[0];
}

void FullyConnectedLayer::resetAccumulatedNablas() {
    accumulatedNablaWeights = arma::zeros(numOutputs, inputHeight * inputWidth * inputDepth);
    accumulatedNablaBiases = arma::zeros(numOutputs);
}

double FullyConnectedLayer::getRandomValueBetweenBorders(int min, int max){
    return ((rand()%(max-min + 1) + min)/(double)(max-min));
}
