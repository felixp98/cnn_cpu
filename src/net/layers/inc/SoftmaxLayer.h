#ifndef SOFTMAXLAYER_H
#define SOFTMAXLAYER_H

#include <iostream>
#include <net/cost/inc/CostFunction.h>
#include <utils/inc/Image.h>
#include "Layer.h"

class SoftmaxLayer : public Layer
{
private:
    size_t numOutputNeurons;

    arma::vec softmaxScores;
    arma::vec softmaxLoss;

    CostFunction* costFunction;
    double cost = 0.0;

    std::vector<Image*> *trainData;
    size_t imageIndex;

    arma::mat weights;
    arma::vec biases;

    arma::mat nablaWeights;
    arma::vec nablaBiases;
    arma::vec deltaError;

    arma::mat accumulatedNablaWeights;
    arma::vec accumulatedNablaBiases;

    arma::vec zWeightedInput;
    arma::vec activationOutput;

public:
    SoftmaxLayer(CostFunction* costFunction, size_t numOutputNeurons);

    void init() override;
    arma::cube& feedForward(arma::cube& input) override;
    void backprop(arma::vec* upstreamGradient) override;
    void init_for_testing(size_t inputHeight, size_t inputWidth, size_t inputDepth);

    void setTrainData(std::vector<Image*> *trainData);
    void setImageIndex(size_t index);

    double getCost() const;

    int getClassOfHighestScore();

    vector<Image *> *getTrainData() const;

    size_t getImageIndex() const;

    const arma::mat &getWeights() const;

    void _resetAccumulatedGradients();
    void updateWeightsAndBiases(size_t batchSize, double learningRate);
    double getRandomValueBetweenBorders(int min, int max);

    void setWeights(const arma::mat &weights);

    void setBiases(const arma::vec &biases);

    const arma::vec &getZWeightedInput() const;

    const arma::vec &getDeltaError() const;

    const arma::mat &getNablaWeights() const;

    const arma::vec &getNablaBiases() const;
};

#endif //SOFTMAXLAYER_H