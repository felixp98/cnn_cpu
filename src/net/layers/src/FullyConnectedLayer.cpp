#include "../inc/FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(){
    std::cout << "new FullyConnectedLayer" << std::endl;
}

void FullyConnectedLayer::init()
{
    std::cout << "init FC" << std::endl;
}

void FullyConnectedLayer::feedForward()
{
    std::cout << "feedforward FC" << std::endl;
}

void FullyConnectedLayer::backprop()
{
    std::cout << "backprop FC" << std::endl;
}