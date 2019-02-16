#include "QuadraticLossLayer.h"

QuadraticLossLayer::QuadraticLossLayer(size_t numInputs)
    : numInputs(numInputs) {
    setType(QUADRATIC_LOSS_LAYER);
}

void QuadraticLossLayer::init() {
    inputHeight = getBeforeLayer()->getOutputHeight();
    inputWidth = getBeforeLayer()->getOutputWidth();
    inputDepth = getBeforeLayer()->getOutputDepth();

    gradientInput = arma::zeros(numInputs, 1, 1);
}

void QuadraticLossLayer::feedForward() {
    this->predictedOutput = arma::vectorise(getBeforeLayer()->getOutput());

    //this->loss = -arma::dot(expectedOutput, arma::log(predictedOutput));
    this->loss = 0.5 * arma::sum(arma::pow((predictedOutput - expectedOutput),2));
}

void QuadraticLossLayer::backPropagate() {
    //arma::vec temp = -(expectedOutput % (1 / predictedOutput));
    arma::vec temp = predictedOutput - expectedOutput;

    gradientInput.slice(0).col(0) = temp;
}

double QuadraticLossLayer::getLoss() const {
    return loss;
}

int QuadraticLossLayer::getMaxIndex() {
    return static_cast<int>(predictedOutput.index_max());
}

