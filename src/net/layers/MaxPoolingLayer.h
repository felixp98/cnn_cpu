#ifndef CNN_GPU_MAXPOOLINGLAYER_H
#define CNN_GPU_MAXPOOLINGLAYER_H

#include <iostream>
#include <armadillo>
#include <cassert>
#include "Layer.h"

class MaxPoolingLayer : public Layer {
private:
    size_t poolingSize;
    size_t stride;

public:
    MaxPoolingLayer(size_t poolingSize, size_t stride);

    void init() override;

    void feedForward() override;

    void backPropagate() override;
};

#endif //CNN_GPU_MAXPOOLINGLAYER_H
