#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#include <vector>
#include <iostream>
#include <armadillo>
#include "Layer.h"

class ConvolutionalLayer : public Layer
{
private:
	arma::cube input;
    arma::cube output;
	size_t inputHeight;
	size_t inputWidth;
	size_t inputDepth;
	size_t numFilters;
	size_t filterSize;
	size_t stride;
	std::vector<arma::cube> filters;

public:
	ConvolutionalLayer(size_t filterSize, size_t numFilters, size_t inputHeight, size_t inputLength, size_t inputDepth);
	
	void init() override;
	void feedForward() override;
	void backprop() override;
};

#endif //CONVOLUTIONALLAYER_H