#ifndef MAXPOOLLAYER_H
#define MAXPOOLLAYER_H

#include <iostream>

class MaxPoolLayer
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
    MaxPoolLayer();

    void init();
    void feedForward();
    void backprop();
};

#endif //MAXPOOLLAYER_H