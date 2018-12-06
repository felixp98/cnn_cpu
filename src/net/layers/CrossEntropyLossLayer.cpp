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
    this->predictedOutput = arma::vectorise(getBeforeLayer()->getOutput());

    this->loss = -arma::dot(expectedOutput,
                            arma::log(predictedOutput));
}

void CrossEntropyLossLayer::backPropagate() {
    arma::vec temp =
            -(expectedOutput % (1 / predictedOutput));
    gradientInput.slice(0).col(0) = temp;
}

double CrossEntropyLossLayer::getLoss() const {
    return loss;
}

int CrossEntropyLossLayer::getMaxIndex() {
    return static_cast<int>(predictedOutput.index_max());
}

