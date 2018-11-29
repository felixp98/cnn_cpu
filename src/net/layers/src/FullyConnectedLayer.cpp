#include <net/cost/inc/QuadraticCost.h>
#include <net/layers/inc/SoftmaxLayer.h>
#include "../inc/FullyConnectedLayer.h"

#define DEBUG false

FullyConnectedLayer::FullyConnectedLayer(ActivationFunction* activationFunction, size_t depth){
    std::cout << "new FullyConnectedLayer" << std::endl;
    this->depth = depth;
    this->activationFunction = activationFunction;

    setType(FULLY_CONNECTED_LAYER);
}

void FullyConnectedLayer::init()
{
    std::cout << "init FC" << std::endl;

    //init in-/outputsize
    if(getBeforeLayer() != nullptr){
        // Hidden Layer
        this->inputHeight = getBeforeLayer()->getOutputHeight();
        this->inputWidth = getBeforeLayer()->getOutputWidth();
        this->inputDepth = getBeforeLayer()->getOutputDepth();

        // if this is the first layer of the network, the input sizes got set directly when the layer was added
    }

    outputHeight = 1;
    outputWidth = 1;
    outputDepth = depth;

    weights = arma::zeros(depth, inputHeight*inputWidth*inputDepth);
    weights.imbue( [&]() { return getRandomValueBetweenBorders(-255, 255); } );

    biases = arma::zeros(depth);
    biases.randu();

    _resetAccumulatedGradients();
}

arma::cube& FullyConnectedLayer::feedForward(arma::cube& input)
{
    this->input = input;

    output = arma::zeros(depth,1,1);
    zWeightedInput = arma::zeros(depth);
    activationOutput = arma::zeros(depth);

    arma::vec vectorisedInput = arma::vectorise(input);
    zWeightedInput = (weights * vectorisedInput) + biases;
    activationOutput = activationFunction->forwardActivation(zWeightedInput);

#if DEBUG
    std::cout << "\nFC:" << std::endl;
    zWeightedInput.print();
    std::cout << "\n" << std::endl;
    activationOutput.print();
#endif

    output.slice(0).col(0) = activationOutput;

    return output;
}

void FullyConnectedLayer::backprop(arma::vec* upstreamGradient)
{
    if(getAfterLayer() == nullptr){
        std::cout << "Backprop error" << std::endl;
        this->upstreamGradient = arma::ones(depth);
        return;
    }

    arma::vec currUpstreamGradient = *upstreamGradient;

    deltaError = arma::zeros(depth);
    nablaWeights = arma::zeros(depth, inputHeight*inputWidth*inputDepth);
    nablaBiases = arma::zeros(depth);

    arma::vec activFunctDerivativesZ = arma::zeros(depth);
    activFunctDerivativesZ = activationFunction->derivativeActivation(zWeightedInput);

    if(getAfterLayer()->getType() == SOFTMAX) {
        deltaError =
                (((SoftmaxLayer*) getAfterLayer())->getWeights().t() * currUpstreamGradient) % activFunctDerivativesZ;
    }else if(getAfterLayer()->getType() == FULLY_CONNECTED_LAYER) {
        deltaError = (((FullyConnectedLayer*) getAfterLayer())->getWeights().t() * currUpstreamGradient) % activFunctDerivativesZ;
    }else{
        std::cout << "Backprop error" << std::endl;
        this->upstreamGradient = arma::ones(depth);
        return;
    }

    nablaWeights = arma::zeros(depth, inputHeight*inputWidth*inputDepth);
    nablaWeights = deltaError * arma::vectorise(input).t();
    nablaBiases = deltaError;

    accumulatedNablaWeights += nablaWeights;
    accumulatedNablaBiases += nablaBiases;

    //weights = weights - nablaWeights * 0.1;
    //biases = biases - nablaBiases * 0.1;

    this->upstreamGradient = deltaError;

#if DEBUG
    std::cout << "\nbp_FC:" << std::endl;
    activFunctDerivativesZ.print();
    std::cout << "\n" << std::endl;
    deltaError.print();
    std::cout << "\n" << std::endl;
    std::cout << weights.n_rows << "x" << weights.n_cols << std::endl;
    std::cout << nablaWeights.n_rows << "x" << nablaWeights.n_cols << std::endl;
    nablaWeights.print();
    std::cout << "\n" << std::endl;
    accumulatedNablaWeights.print();
#endif
}

void FullyConnectedLayer::_resetAccumulatedGradients()
{
    accumulatedNablaWeights = arma::zeros(depth, inputHeight*inputWidth*inputDepth);
    accumulatedNablaBiases = arma::zeros(depth);
}

void FullyConnectedLayer::updateWeightsAndBiases(size_t batchSize, double learningRate)
{
    weights = weights - (learningRate/batchSize) * accumulatedNablaWeights;
    biases = biases - (learningRate/batchSize) * accumulatedNablaBiases;

    _resetAccumulatedGradients();
}

arma::mat &FullyConnectedLayer::getWeights() {
    return weights;
}

double FullyConnectedLayer::getRandomValueBetweenBorders(int min, int max){
    return ((rand()%(max-min + 1) + min)/(double)(max-min));
}

arma::mat &FullyConnectedLayer::getNablaWeights() {
    return nablaWeights;
}

void FullyConnectedLayer::setWeights(arma::mat &weights) {
    FullyConnectedLayer::weights = weights;
}

void FullyConnectedLayer::init_for_testing(size_t inputHeight, size_t inputWidth, size_t inputDepth)
{
    this->inputHeight = inputHeight;
    this->inputWidth = inputWidth;
    this->inputDepth = inputDepth;

    outputHeight = 1;
    outputWidth = 1;
    outputDepth = depth;

    weights = arma::zeros(depth, inputHeight*inputWidth*inputDepth);
    weights.imbue( [&]() { return getRandomValueBetweenBorders(-255, 255); } );

    //weights.print();

    biases = arma::zeros(depth);
    biases.randu();

    _resetAccumulatedGradients();
}

void FullyConnectedLayer::setBiases(arma::vec &biases) {
    FullyConnectedLayer::biases = biases;
}

const arma::vec &FullyConnectedLayer::getZWeightedInput() const {
    return zWeightedInput;
}

const arma::vec &FullyConnectedLayer::getDeltaError() const {
    return deltaError;
}

#undef DEBUG