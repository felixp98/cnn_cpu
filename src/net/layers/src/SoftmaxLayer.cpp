#include "../inc/SoftmaxLayer.h"

SoftmaxLayer::SoftmaxLayer(size_t numOutputNeurons){
    std::cout << "new SoftmaxLayer" << std::endl;
    this->numOutputNeurons = numOutputNeurons;
}

void SoftmaxLayer::init()
{
    std::cout << "init softmax" << std::endl;

    //init in-/outputsize
    if(getBeforeLayer() != nullptr){
        // Hidden Layer
        this->inputHeight = getBeforeLayer()->getOutputHeight();
        this->inputWidth = getBeforeLayer()->getOutputWidth();
        this->inputDepth = getBeforeLayer()->getOutputDepth();
    }

    outputHeight = 1;
    outputWidth = 1;
    outputDepth = numOutputNeurons;
}

arma::cube& SoftmaxLayer::feedForward(arma::cube& input)
{
    //set input if layer is hidden-layer
    if(getBeforeLayer() != nullptr){
        input = getBeforeLayer()->getOutput();
    }

    output = arma::zeros(numOutputNeurons,1,1);

    double expSum = arma::accu(arma::exp(arma::vectorise(input) - arma::max(arma::vectorise(input))));
    std::cout << expSum << std::endl;
    output.slice(0) = arma::exp(arma::vectorise(input) - arma::max(arma::vectorise(input)))/expSum;

    output.print();
}

void SoftmaxLayer::backprop()
{
    std::cout << "backprop softmax" << std::endl;
}