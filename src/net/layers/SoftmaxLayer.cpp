#include "SoftmaxLayer.h"
#include <chrono>

#define TIME_MEASURE true

#if TIME_MEASURE
using namespace std::chrono;
#endif

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
#if TIME_MEASURE
    high_resolution_clock::time_point t_forward_start = high_resolution_clock::now();
#endif

    this->input = getBeforeLayer()->getOutput();
    arma::vec flatInput = arma::vectorise(input);

    //flatInput.print();

    double sumExp = arma::accu(arma::exp(flatInput - arma::max(flatInput)));
    arma::vec temp = arma::exp(flatInput - arma::max(flatInput)) / sumExp;
    //temp.print();
    output.slice(0).col(0) = temp;

#if TIME_MEASURE
    high_resolution_clock::time_point t_forward_stop = high_resolution_clock::now();
    forwardDuration += duration_cast<microseconds>(t_forward_stop-t_forward_start).count();
#endif
}

void SoftmaxLayer::backPropagate() {
#if TIME_MEASURE
    high_resolution_clock::time_point t_backward_start = high_resolution_clock::now();
#endif

    arma::vec upstreamGradient = arma::vectorise(getAfterLayer()->getGradientInput());
    arma::vec flatOutput = arma::vectorise(output);

    double sub = arma::dot(upstreamGradient, flatOutput);
    arma::vec temp = (upstreamGradient - sub) % flatOutput;
    gradientInput.slice(0).col(0) = temp;

#if TIME_MEASURE
    high_resolution_clock::time_point t_backward_stop = high_resolution_clock::now();
    backwardDuration += duration_cast<microseconds>(t_backward_stop-t_backward_start).count();
#endif
}

