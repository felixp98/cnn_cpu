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

arma::cube& FullyConnectedLayer::feedForward(arma::cube& input)
{
    this->input = input;

    output = arma::zeros(depth,1,1);
    arma::vec vOutput = arma::zeros(depth);

    arma::vec vectorisedInput = arma::vectorise(input);
    vOutput = (weights * vectorisedInput) + biases;

    output.slice(0).col(0) = vOutput;

    output /= 100;


    return output;
}

void FullyConnectedLayer::backprop()
{
    std::cout << "backprop FC" << std::endl;
}