#include "../inc/FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(size_t depth){
    std::cout << "new FullyConnectedLayer" << std::endl;
    this->depth = depth;
}

void FullyConnectedLayer::init()
{
    std::cout << "init FC" << std::endl;

    //init in-/outputsize
    if(getBeforeLayer() != nullptr){
        // Hidden Layer
        this->inputHeight = getBeforeLayer()->getOutputHeight();
        this->inputWidth = getBeforeLayer()->getOutputWidth();
        this->inputDepth = getBeforeLayer()->getOutputDepth();

        // if this is the first layer of the network, the input sizes got set directly when the layer was added
    }

    outputHeight = 1;
    outputWidth = 1;
    outputDepth = depth;

    weights = arma::zeros(depth, inputHeight*inputWidth*inputDepth);
    weights.randu();

    biases = arma::zeros(depth);
}

void FullyConnectedLayer::feedForward()
{
    //set input if layer is hidden-layer
    if(getBeforeLayer() != nullptr){
        input = getBeforeLayer()->getOutput();
    }

    output = arma::zeros(depth,1,1);

    arma::vec vectorisedInput = arma::vectorise(input);
    output.slice(0) = (weights * vectorisedInput) + biases;

    output.print();
}

void FullyConnectedLayer::backprop()
{
    std::cout << "backprop FC" << std::endl;
}