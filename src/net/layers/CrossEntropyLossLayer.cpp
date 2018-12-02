//
// Created by felix on 02.12.18.
//

#include "CrossEntropyLossLayer.h"

CrossEntropyLossLayer::CrossEntropyLossLayer(size_t numInputs)
    : numInputs(numInputs) {
    setType(CROSS_ENTROPY_COST_LAYER);
}

void CrossEntropyLossLayer::init() {
    inputHeight = getBeforeLayer()->getOutputHeight();
    inputWidth = getBeforeLayer()->getOutputWidth();
    inputDepth = getBeforeLayer()->getOutputDepth();

    gradientInput = arma::zeros(numInputs, 1, 1);
}

void CrossEntropyLossLayer::feedForward() {
    this->predictedDistribution = arma::vectorise(getBeforeLayer()->getOutput());

    // Compute the loss and cache that too.
    //expectedOutput.print();
    //predictedDistribution.print();
    this->loss = -arma::dot(expectedOutput,
                            arma::log(predictedDistribution));
}

void CrossEntropyLossLayer::backPropagate() {
    arma::vec temp =
            -(expectedOutput % (1 / predictedDistribution));
    gradientInput.slice(0).col(0) = temp;
}

double CrossEntropyLossLayer::getLoss() const {
    return loss;
}

int CrossEntropyLossLayer::getMaxIndex() {
    return static_cast<int>(predictedDistribution.index_max());
}

