#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include <iostream>
#include "Layer.h"

class SoftmaxLayer : public Layer
{
private:


public:
    SoftmaxLayer();

    void init() override;
    void feedForward() override;
    void backprop() override;
};

#endif //SOFTMAXLAYER_H