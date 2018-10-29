#include "../inc/ConvolutionalLayer.h"
#include <armadillo>
#include <cmath>
#include <vector>
#include <net/layers/inc/ConvolutionalLayer.h>


ConvolutionalLayer::ConvolutionalLayer(size_t filterSize, size_t numFilters, size_t inputHeight, size_t inputWidth, size_t inputDepth){
	std::cout << "new ConvLayer" << std::endl;
	this->filterSize = filterSize;
	this->numFilters = numFilters;
	this->inputHeight = inputHeight;
	this->inputWidth = inputWidth;
	this->inputDepth = inputDepth;
}

void ConvolutionalLayer::init()
{
	std::cout << "init conv" << std::endl;
    //init filters
    filters.resize(numFilters);
    for(int i=0; i<numFilters; ++i){
        filters[i] = arma::zeros(filterSize, filterSize, inputDepth);
        filters[i].fill(0.5);
    }

}

void ConvolutionalLayer::feedForward()
{
	std::cout << "feedforward conv" << std::endl;

    // Output initialization.
    this->output = arma::zeros((inputHeight - filterSize)/stride + 1,
                         (inputWidth - filterSize)/stride + 1,
                         numFilters);

    std::cout << "initialized output cube" << std::endl;

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

    this->input = input;
    this->output = output;

    std::cout << "input[0]: " << input[0] << std::endl;

}

void ConvolutionalLayer::backprop()
{
	std::cout << "backprop conv" << std::endl;
}
