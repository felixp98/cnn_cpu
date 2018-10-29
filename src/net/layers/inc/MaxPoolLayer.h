#ifndef MAXPOOLLAYER_H
#define MAXPOOLLAYER_H

#include <iostream>
#include "Layer.h"

class MaxPoolLayer : public Layer
{
private:

public:
    MaxPoolLayer();

    void init() override;
    void feedForward() override;
    void backprop() override;
};

#endif //MAXPOOLLAYER_H