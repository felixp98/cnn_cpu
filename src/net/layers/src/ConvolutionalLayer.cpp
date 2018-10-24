#include "../inc/ConvolutionalLayer.h"

ConvolutionalLayer::ConvolutionalLayer(int filterSize, int numFilters){
	std::cout << "new ConvLayer" << std::endl;
}

void ConvolutionalLayer::init()
{
	std::cout << "init conv" << std::endl;
}

void ConvolutionalLayer::feedForward()
{
	std::cout << "feedforward conv" << std::endl;
}

void ConvolutionalLayer::backprop()
{
	std::cout << "backprop conv" << std::endl;
}