#ifndef MAXPOOLLAYER_H
#define MAXPOOLLAYER_H

#include <iostream>
#include "Layer.h"

class MaxPoolLayer : public Layer
{
private:
    size_t poolingSize;
    size_t stride;

public:
    MaxPoolLayer(size_t poolingSize, size_t stride);

    void init() override;
    arma::cube& feedForward(arma::cube& input) override;
    void backprop() override;
    void init_for_testing(size_t inputHeight, size_t inputWidth, size_t inputDepth);
};

#endif //MAXPOOLLAYER_H