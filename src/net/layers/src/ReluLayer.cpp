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
    std::cout << "feedforward relu" << std::endl;
}

void ReluLayer::backprop()
{
    std::cout << "backprop relu" << std::endl;
}