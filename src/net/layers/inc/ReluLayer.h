#ifndef RELULAYER_H
#define RELULAYER_H

#include <iostream>
#include "Layer.h"

class ReluLayer : public Layer
{
private:


public:
    ReluLayer();

    void init() override;
    arma::cube& feedForward(arma::cube& input) override;
    void backprop() override;
};

#endif //RELULAYER_H