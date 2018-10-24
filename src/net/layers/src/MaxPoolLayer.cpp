#include "../inc/MaxPoolLayer.h"

MaxPoolLayer::MaxPoolLayer(){
    std::cout << "new MaxPoolLayer" << std::endl;
}

void MaxPoolLayer::init()
{
    std::cout << "init maxpool" << std::endl;
}

void MaxPoolLayer::feedForward()
{
    std::cout << "feedforward MaxPool" << std::endl;
}

void MaxPoolLayer::backprop()
{
    std::cout << "backprop maxpool" << std::endl;
}