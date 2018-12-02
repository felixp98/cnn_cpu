//
// Created by felix on 02.12.18.
//

#include "ConvolutionalLayer.h"

ConvolutionalLayer::ConvolutionalLayer(size_t numFilters, size_t filterSize, size_t stride)
    : numFilters(numFilters), filterSize(filterSize), stride(stride) {
    setType(CONV_LAYER);
}

void ConvolutionalLayer::init() {
    this->inputHeight = getBeforeLayer()->getOutputHeight();
    this->inputWidth = getBeforeLayer()->getOutputWidth();
    this->inputDepth = getBeforeLayer()->getOutputDepth();

    outputHeight = (inputHeight - filterSize) / stride + 1;
    outputWidth = (inputWidth - filterSize) / stride + 1;
    outputDepth = numFilters;

    //std::cout << "Input Conv: " << inputHeight << "x" << inputWidth << "x" << inputDepth << std::endl;
    std::cout << "Output Conv: " << outputHeight << "x" << outputWidth << "x" << outputDepth << std::endl;

    // Initialize the filters.
    filters.resize(numFilters);
    for (size_t i = 0; i < numFilters; i++) {
        filters[i] = arma::zeros(filterSize, filterSize, inputDepth);
        filters[i].imbue([&]() { return _getTruncNormalVal(0.0, 1.0); });
    }

    _resetAccumulatedGradients();
}

void ConvolutionalLayer::feedForward() {
    this->input = getBeforeLayer()->getOutput();

    // Output cube initialization.
    this->output = arma::zeros(outputHeight, outputWidth, outputDepth);

    // Perform convolution for each filter.
    for (size_t fidx = 0; fidx < numFilters; fidx++)
    {
        for (size_t i=0; i <= inputHeight - filterSize; i += stride)
            for (size_t j=0; j <= inputWidth - filterSize; j += stride)
                output((i/stride), (j/stride), fidx) = arma::dot(
                        arma::vectorise(
                                input.subcube(i, j, 0,
                                              i+filterSize-1, j+filterSize-1, inputDepth-1)
                        ),
                        arma::vectorise(filters[fidx]));
    }
}

void ConvolutionalLayer::backPropagate() {
    arma::cube upstreamGradient = getAfterLayer()->getGradientInput();

    // Initialize gradient wrt input. Note that the dimensions are same as those
    // of the input.
    gradientInput = arma::zeros(arma::size(input));

    // Compute the gradient wrt input.
    for (size_t sidx = 0; sidx < numFilters; sidx++) {
        for (size_t r = 0; r < output.n_rows; r++) {
            for (size_t c = 0; c < output.n_cols; c++) {
                arma::cube tmp(arma::size(input), arma::fill::zeros);
                tmp.subcube(r * stride,
                            c * stride,
                            0,
                            (r * stride) + filterSize - 1,
                            (c * stride) + filterSize - 1,
                            inputDepth - 1)
                        = filters[sidx];
                gradientInput += upstreamGradient.slice(sidx)(r, c) * tmp;
            }
        }
    }

    // Initialize the gradient wrt filters.
    gradFilters.clear();
    gradFilters.resize(numFilters);
    for (size_t i = 0; i < numFilters; i++)
        gradFilters[i] = arma::zeros(filterSize, filterSize, inputDepth);

    // Compute the gradient wrt filters.
    for (size_t fidx = 0; fidx < numFilters; fidx++) {
        for (size_t r = 0; r < output.n_rows; r++) {
            for (size_t c = 0; c < output.n_cols; c++) {
                arma::cube tmp(arma::size(filters[fidx]), arma::fill::zeros);
                tmp = input.subcube(r * stride,
                                    c * stride,
                                    0,
                                    (r * stride) + filterSize - 1,
                                    (c * stride) + filterSize - 1,
                                    inputDepth - 1);
                gradFilters[fidx] += upstreamGradient.slice(fidx)(r, c) * tmp;
            }
        }
    }

    // Update the accumulated gradient wrt filters.
    for (size_t fidx = 0; fidx < numFilters; fidx++)
        accumulatedGradFilters[fidx] += gradFilters[fidx];
}

double ConvolutionalLayer::_getTruncNormalVal(double mean, double variance) {
    double stddev = sqrt(variance);
    arma::mat candidate = {3.0 * stddev};
    while (std::abs(candidate[0] - mean) > 2.0 * stddev)
        candidate.randn(1, 1);
    return candidate[0];
}

void ConvolutionalLayer::_resetAccumulatedGradients() {
    accumulatedGradFilters.clear();
    accumulatedGradFilters.resize(numFilters);
    for (size_t fidx = 0; fidx < numFilters; fidx++)
        accumulatedGradFilters[fidx] = arma::zeros(filterSize,
                                                   filterSize,
                                                   inputDepth);
}

std::vector<arma::cube> ConvolutionalLayer::getGradientWrtFilters() {
    return gradFilters;
}

void ConvolutionalLayer::UpdateFilterWeights(size_t batchSize, double learningRate) {
    for (size_t fidx = 0; fidx < numFilters; fidx++)
        filters[fidx] -= learningRate * (accumulatedGradFilters[fidx] / batchSize);

    _resetAccumulatedGradients();
}

void ConvolutionalLayer::setFilters(std::vector<arma::cube> filters) {
    this->filters = filters;
}

std::vector<arma::cube> ConvolutionalLayer::getFilters() {
    return this->filters;
}
