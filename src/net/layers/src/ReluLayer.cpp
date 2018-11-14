#include "../inc/ReluLayer.h"

ReluLayer::ReluLayer(){
    std::cout << "new ReluLayer" << std::endl;
}

void ReluLayer::init()
{
    std::cout << "init relu" << std::endl;

    //init in-/outputsize
    if(getBeforeLayer() != nullptr){
        // Hidden Layer
        this->inputHeight = getBeforeLayer()->getOutputHeight();
        this->inputWidth = getBeforeLayer()->getOutputWidth();
        this->inputDepth = getBeforeLayer()->getOutputDepth();

        // if this is the first layer of the network, the input sizes got set directly when the layer was added
    }

    outputHeight = inputHeight;
    outputWidth = inputWidth;
    outputDepth = inputDepth;
}

arma::cube& ReluLayer::feedForward(arma::cube& input)
{
    output = arma::zeros(input.n_rows, input.n_cols, input.n_slices);

    arma::cube::iterator it_input = input.begin();
    arma::cube::iterator it_input_end = input.end();
    arma::cube::iterator it_output = output.begin();

    for(; it_input != it_input_end; ++it_input)
    {
        if(*it_input < 0){
            *it_output = 0;
        }else{
            *it_output = *it_input;
        }
        ++it_output;
    }

    return output;
}

void ReluLayer::backprop(arma::vec& upstreamGradient)
{
    std::cout << "backprop relu" << std::endl;
}