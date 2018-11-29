#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#include <vector>
#include <iostream>
#include <armadillo>
#include "Layer.h"

class ConvolutionalLayer : public Layer
{
private:
	size_t numFilters;
	size_t filterSize;
	size_t stride;

	std::vector<arma::cube> filters;

public:
	ConvolutionalLayer(size_t numFilters, size_t filterSize, size_t stride);
	
	void init() override;
	arma::cube& feedForward(arma::cube& input) override;
	void backprop(arma::vec* upstreamGradient) override;
	void init_for_testing(size_t inputHeight, size_t inputWidth, size_t inputDepth,
                          std::vector<arma::cube>& filters);

	//only for testing:
	arma::cube feedForwardTesting(arma::cube input, std::vector<arma::cube> filter);
};

#endif //CONVOLUTIONALLAYER_H