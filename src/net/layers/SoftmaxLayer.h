#ifndef CNN_GPU_SOFTMAXLAYER_H
#define CNN_GPU_SOFTMAXLAYER_H

#include <iostream>
#include <armadillo>
#include "Layer.h"

class SoftmaxLayer : public Layer {
private:
    size_t numInputs;

public:
    explicit SoftmaxLayer(size_t numInputs);

    void init() override;

    void feedForward() override;

    void backPropagate() override;
};

#endif //CNN_CPU_SOFTMAXLAYER_H
