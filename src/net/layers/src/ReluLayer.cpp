#include "../inc/ReluLayer.h"

ReluLayer::ReluLayer(){
    std::cout << "new ReluLayer" << std::endl;
}

void ReluLayer::init()
{
    std::cout << "init relu" << std::endl;
}

arma::cube& ReluLayer::feedForward(arma::cube& input)
{
    output = arma::zeros(arma::size(input));
    output = arma::max(input, output);

    return output;
}

void ReluLayer::backprop()
{
    std::cout << "backprop relu" << std::endl;
}