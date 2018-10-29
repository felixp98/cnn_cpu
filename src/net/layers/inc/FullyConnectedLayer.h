#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include <iostream>
#include "Layer.h"

class FullyConnectedLayer : public Layer
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

    void init() override;
    void feedForward() override;
    void backprop() override;
};

#endif //FULLYCONNECTEDLAYER_H