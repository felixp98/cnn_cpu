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

    arma::mat nablaWeights;
    arma::vec nablaBiases;

    arma::mat accumulatedNablaWeights;
    arma::vec accumulatedNablaBiases;

    double getRandomWeight(double mean, double variance);
    void resetAccumulatedNablas();

public:
    explicit FullyConnectedLayer(size_t numOutputs);

    void init() override;

    void feedForward() override;

    void backPropagate() override;

    void updateWeightsAndBiases(size_t batchSize, double learningRate);

    double getRandomValueBetweenBorders(int min, int max);
};

#endif //CNN_GPU_FULLYCONNECTEDLAYER_H
