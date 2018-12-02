//
// Created by felix on 02.12.18.
//

#ifndef CNN_GPU_INPUTLAYER_H
#define CNN_GPU_INPUTLAYER_H

#include <iostream>
#include <armadillo>
#include "Layer.h"

class InputLayer : public Layer {
public:
    InputLayer();

    void init() override;

    void feedForward() override;

    void backPropagate() override;
};

#endif //CNN_GPU_INPUTLAYER_H
