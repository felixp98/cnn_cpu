//
// Created by felix on 02.12.18.
//

#ifndef CNN_GPU_FULLYCONNECTEDLAYER_H
#define CNN_GPU_FULLYCONNECTEDLAYER_H

#include <armadillo>
#include <vector>
#include <cmath>
#include <cassert>
#include "Layer.h"

class FullyConnectedLayer : public Layer {
private:
    size_t numOutputs;

    arma::mat weights;
    arma::vec biases;

    arma::mat gradWeights;
    arma::vec gradBiases;

    arma::mat accumulatedGradWeights;
    arma::vec accumulatedGradBiases;

    double _getTruncNormalVal(double mean, double variance);
    void _resetAccumulatedGradients();

public:
    explicit FullyConnectedLayer(size_t numOutputs);

    void init() override;

    void feedForward() override;

    void backPropagate() override;

    void UpdateWeightsAndBiases(size_t batchSize, double learningRate);

    double getRandomValueBetweenBorders(int min, int max);
};

#endif //CNN_GPU_FULLYCONNECTEDLAYER_H
