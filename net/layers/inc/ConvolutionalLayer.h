#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#include <iostream>

class ConvolutionalLayer
{
private:
	double input;
	double output;
	int inputHeight;
	int inputWidth;
	int inputDepth;
	int numFilters;
	int filterSize;
	int stride;
	double filters;

public:
	ConvolutionalLayer(int filterSize, int numFilters);
	
	void init();
	void feedForward();
	void backprop();
};

#endif //CONVOLUTIONALLAYER_H