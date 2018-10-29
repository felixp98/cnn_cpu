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
    void feedForward() override;
    void backprop() override;
};

#endif //RELULAYER_H