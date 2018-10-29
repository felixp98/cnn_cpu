#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include <iostream>
#include "Layer.h"

class FullyConnectedLayer : public Layer
{
private:


public:
    FullyConnectedLayer();

    void init() override;
    void feedForward() override;
    void backprop() override;
};

#endif //FULLYCONNECTEDLAYER_H