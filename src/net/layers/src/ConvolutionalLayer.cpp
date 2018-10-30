#include <armadillo>
#include <cmath>
#include <vector>
#include <net/layers/inc/ConvolutionalLayer.h>


ConvolutionalLayer::ConvolutionalLayer(size_t numFilters, size_t filterSize, size_t stride){
	std::cout << "new ConvLayer" << std::endl;

    this->numFilters = numFilters;
	this->filterSize = filterSize;
	this->stride = stride;
}

void ConvolutionalLayer::init()
{
	std::cout << "init conv" << std::endl;

	//init in-/outputsize
	if(getBeforeLayer() != nullptr){
	    // Hidden Layer
        this->inputHeight = getBeforeLayer()->getInputHeight();
        this->inputWidth = getBeforeLayer()->getInputWidth();
        this->inputDepth = getBeforeLayer()->getInputDepth();
	}

    //init filters
    filters.resize(numFilters);
    for(int i=0; i<numFilters; ++i){
        filters[i] = arma::zeros(filterSize, filterSize, inputDepth);
        filters[i].fill(0.5); //Todo: fill with random values
    }

    //Todo: remove on target
    this->input = arma::zeros(32, 32, 3);
    this->input.fill(0.4);

}

void ConvolutionalLayer::feedForward()
{
	std::cout << "feedforward conv" << std::endl;

    // Output initialization.
    this->output = arma::zeros((inputHeight - filterSize)/stride + 1,
                         (inputWidth - filterSize)/stride + 1,
                         numFilters);

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

void ConvolutionalLayer::backprop()
{
	std::cout << "backprop conv" << std::endl;
}
