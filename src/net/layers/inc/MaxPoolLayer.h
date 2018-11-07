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
};

#endif //MAXPOOLLAYER_H