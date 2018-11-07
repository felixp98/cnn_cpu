#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include <iostream>
#include "Layer.h"

class SoftmaxLayer : public Layer
{
private:
    size_t numOutputNeurons;

public:
    SoftmaxLayer(size_t numOutputNeurons);

    void init() override;
    arma::cube& feedForward(arma::cube& input) override;
    void backprop() override;
};

#endif //SOFTMAXLAYER_H