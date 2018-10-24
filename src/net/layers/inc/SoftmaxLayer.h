#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include <iostream>

class SoftmaxLayer
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
    SoftmaxLayer();

    void init();
    void feedForward();
    void backprop();
};

#endif //SOFTMAXLAYER_H