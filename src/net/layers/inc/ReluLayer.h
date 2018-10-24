#ifndef RELULAYER_H
#define RELULAYER_H

#include <iostream>

class ReluLayer
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
    ReluLayer();

    void init();
    void feedForward();
    void backprop();
};

#endif //RELULAYER_H