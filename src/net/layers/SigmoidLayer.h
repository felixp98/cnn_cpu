#ifndef CNN_GPU_SIGMOIDLAYER_H
#define CNN_GPU_SIGMOIDLAYER_H

#include <iostream>
#include <armadillo>
#include "Layer.h"

class SigmoidLayer : public Layer {
public:
    SigmoidLayer();

    void init() override;

    void feedForward() override;

    void backPropagate() override;
};

#endif //CNN_GPU_SIGMOIDLAYER_H
