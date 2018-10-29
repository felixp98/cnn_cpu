#ifndef MAXPOOLLAYER_H
#define MAXPOOLLAYER_H

#include <iostream>
#include "Layer.h"

class MaxPoolLayer : public Layer
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

    void init() override;
    void feedForward() override;
    void backprop() override;
};

#endif //MAXPOOLLAYER_H