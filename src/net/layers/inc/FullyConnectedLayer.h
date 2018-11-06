#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include <iostream>
#include "Layer.h"

class FullyConnectedLayer : public Layer
{
private:
    arma::mat weights;
    arma::vec biases;
    size_t depth;

public:
    FullyConnectedLayer(size_t depth);

    void init() override;
    void feedForward() override;
    void backprop() override;
};

#endif //FULLYCONNECTEDLAYER_H