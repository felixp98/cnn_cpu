#ifndef CNN_CPU_QUADRATICLOSSLAYER_H
#define CNN_CPU_QUADRATICLOSSLAYER_H

#include <iostream>
#include <cassert>
#include <armadillo>
#include "Layer.h"

class QuadraticLossLayer : public Layer {
private:
    size_t numInputs;
    arma::vec predictedOutput;

    double loss;

public:
    explicit QuadraticLossLayer(size_t numInputs);

    void init() override;

    void feedForward() override;

    void backPropagate() override;

    double getLoss() const;

    int getMaxIndex();
};

#endif //CNN_CPU_QUADRATICLOSSLAYER_H
