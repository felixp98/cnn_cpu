//
// Created by felix on 02.12.18.
//

#ifndef CNN_GPU_RELULAYER_H
#define CNN_GPU_RELULAYER_H

#include <iostream>
#include <armadillo>
#include "Layer.h"

class ReluLayer : public Layer {
public:
    ReluLayer();

    void init() override;

    void feedForward() override;

    void backPropagate() override;
};

#endif //CNN_GPU_RELULAYER_H
