#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include <iostream>
#include "Layer.h"

class SoftmaxLayer : public Layer
{
private:
    size_t numOutputNeurons;

    arma::vec softmaxScores;
    arma::vec softmaxLoss;

public:
    SoftmaxLayer(size_t numOutputNeurons);

    void init() override;
    arma::cube& feedForward(arma::cube& input) override;
    void backprop(arma::vec& upstreamGradient) override;
    void init_for_testing(size_t inputHeight, size_t inputWidth, size_t inputDepth);
};

#endif //SOFTMAXLAYER_H