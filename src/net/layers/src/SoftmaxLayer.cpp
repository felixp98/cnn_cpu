#include "../inc/SoftmaxLayer.h"
#include <math.h>

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

    softmaxScores = arma::zeros(numOutputNeurons);
    softmaxLoss = arma::zeros(numOutputNeurons);
}

arma::cube& SoftmaxLayer::feedForward(arma::cube& input)
{
    this->input = input;

    output = arma::zeros(numOutputNeurons,1,1);

    arma::vec vectorisedInput = arma::vectorise(input);
    double sumExp = 0.0;
    for(size_t i = 0; i< vectorisedInput.size(); i++){
        sumExp += exp(vectorisedInput.at(i));
    }

    for(size_t i = 0; i< softmaxScores.size(); i++){
        softmaxScores(i) = exp(vectorisedInput.at(i))/sumExp;
        softmaxLoss(i) = -log(softmaxScores(i));
    }

    output.slice(0).col(0) = softmaxScores;

    output.print();
    softmaxLoss.print();

    return output;
}

void SoftmaxLayer::backprop(arma::vec& upstreamGradient)
{
    std::cout << "backprop softmax" << std::endl;
}

void SoftmaxLayer::init_for_testing(size_t inputHeight, size_t inputWidth, size_t inputDepth){
    this->inputHeight = inputHeight;
    this->inputWidth = inputWidth;
    this->inputDepth = inputDepth;

    outputHeight = 1;
    outputWidth = 1;
    outputDepth = numOutputNeurons;
}