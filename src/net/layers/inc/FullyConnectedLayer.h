#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include <iostream>
#include <net/activation/inc/ActivationFunction.h>
#include "Layer.h"

class FullyConnectedLayer : public Layer
{
private:
    ActivationFunction* activationFunction;

    arma::mat weights;
    arma::vec biases;
    size_t depth;

    arma::mat nablaWeights;
    arma::vec nablaBiases;
    arma::vec deltaError;

    arma::mat accumulatedNablaWeights;
    arma::vec accumulatedNablaBiases;

    arma::vec zWeightedInput;
    arma::vec activationOutput;

public:
    FullyConnectedLayer(ActivationFunction* activationFunction, size_t depth);

    void init() override;
    arma::cube& feedForward(arma::cube& input) override;
    void backprop(arma::vec* upstreamGradient) override;

    void _resetAccumulatedGradients();
    void updateWeightsAndBiases(size_t batchSize, double learningRate);

    arma::mat &getWeights();

    arma::mat &getNablaWeights();

    void setWeights(arma::mat &weights);

    void setBiases(arma::vec &biases);

    const arma::vec &getZWeightedInput() const;

    const arma::vec &getDeltaError() const;

    double getRandomValueBetweenBorders(int min, int max);

    void init_for_testing(size_t inputHeight, size_t inputWidth, size_t inputDepth);
};

#endif //FULLYCONNECTEDLAYER_H