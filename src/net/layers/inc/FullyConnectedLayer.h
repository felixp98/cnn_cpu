#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include <iostream>
#include "Layer.h"

class FullyConnectedLayer : public Layer
{
private:
    arma::mat weights;
    arma::vec biases;
    size_t depth;

    arma::cube gradInput;
    arma::mat gradWeights;
    arma::vec gradBiases;

    arma::cube accumulatedGradInput;
    arma::mat accumulatedGradWeights;
    arma::vec accumulatedGradBiases;

public:
    explicit FullyConnectedLayer(size_t depth);

    void init() override;
    arma::cube& feedForward(arma::cube& input) override;
    void backprop(arma::vec& upstreamGradient) override;

    void _resetAccumulatedGradients();
    void UpdateWeightsAndBiases(size_t batchSize, double learningRate);
};

#endif //FULLYCONNECTEDLAYER_H