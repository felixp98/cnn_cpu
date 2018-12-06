#ifndef CNN_GPU_CROSSENTROPYLOSSLAYER_H
#define CNN_GPU_CROSSENTROPYLOSSLAYER_H

#include <iostream>
#include <cassert>
#include <armadillo>
#include "Layer.h"

class CrossEntropyLossLayer : public Layer {
private:
    size_t numInputs;
    arma::vec predictedOutput;

    double loss;

public:
    explicit CrossEntropyLossLayer(size_t numInputs);

    void init() override;

    void feedForward() override;

    void backPropagate() override;

    double getLoss() const;

    int getMaxIndex();
};

#endif //CNN_GPU_CROSSENTROPYLOSSLAYER_H
