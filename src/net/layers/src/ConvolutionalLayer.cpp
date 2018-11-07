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
        this->inputHeight = getBeforeLayer()->getOutputHeight();
        this->inputWidth = getBeforeLayer()->getOutputWidth();
        this->inputDepth = getBeforeLayer()->getOutputDepth();

        // if this is the first layer of the network, the input sizes got set directly when the layer was added
	}

	outputHeight = (inputHeight - filterSize)/stride + 1;
	outputWidth = (inputWidth - filterSize)/stride + 1;
	outputDepth = numFilters;

    //init filters
    filters.resize(numFilters);
    for(int i=0; i<numFilters; ++i){
        filters[i] = arma::zeros(filterSize, filterSize, inputDepth);
        filters[i].randu();
    }
}

arma::cube& ConvolutionalLayer::feedForward(arma::cube& input)
{
    this->input = input;

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

    return output;
}

void ConvolutionalLayer::backprop()
{
	std::cout << "backprop conv" << std::endl;
}

/*arma::cube ConvolutionalLayer::feedForwardTesting(arma::cube input, std::vector<arma::cube> filter)
{
    this->input = input;
    this->filters = filter;

    inputHeight = input.n_rows;
    inputWidth = input.n_cols;
    inputDepth = input.n_slices;

    outputHeight = (inputHeight - filterSize)/stride + 1;
    outputWidth = (inputWidth - filterSize)/stride + 1;
    outputDepth = numFilters;

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

    return output;
}*/

void ConvolutionalLayer::init_for_testing(size_t inputHeight, size_t inputWidth, size_t inputDepth,
                                            std::vector<arma::cube>& filters)
{
    this->inputHeight = inputHeight;
    this->inputWidth = inputWidth;
    this->inputDepth = inputDepth;

    outputHeight = (inputHeight - filterSize)/stride + 1;
    outputWidth = (inputWidth - filterSize)/stride + 1;
    outputDepth = numFilters;

    this->filters = std::move(filters);
}