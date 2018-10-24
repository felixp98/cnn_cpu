#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include <iostream>

class FullyConnectedLayer
{
private:
    double input;
    double output;
    int inputHeight;
    int inputWidth;
    int inputDepth;
    int numFilters;
    int filterSize;
    int stride;
    double filters;

public:
    FullyConnectedLayer();

    void init();
    void feedForward();
    void backprop();
};

#endif //FULLYCONNECTEDLAYER_H