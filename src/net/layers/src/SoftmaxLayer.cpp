#include "../inc/SoftmaxLayer.h"
#include <math.h>
#include <net/layers/inc/SoftmaxLayer.h>
#include <net/activation/inc/SigmoidActivation.h>

#define DEBUG false

SoftmaxLayer::SoftmaxLayer(CostFunction* costFunction, size_t numOutputNeurons){
    std::cout << "new SoftmaxLayer" << std::endl;
    this->numOutputNeurons = numOutputNeurons;
    this->costFunction = costFunction;

    setType(SOFTMAX);
}

void SoftmaxLayer::init()
{
    std::cout << "init softmax" << std::endl;

    //init in-/outputsize
    if(getBeforeLayer() != nullptr){
        // Hidden Layer
        this->inputHeight = getBeforeLayer()->getOutputHeight();
        this->inputWidth = getBeforeLayer()->getOutputWidth();
        this->inputDepth = getBeforeLayer()->getOutputDepth();
    }

    outputHeight = 1;
    outputWidth = 1;
    outputDepth = numOutputNeurons;

    softmaxScores = arma::zeros(numOutputNeurons);
    //softmaxLoss = arma::zeros(numOutputNeurons);

    weights = arma::zeros(numOutputNeurons, inputHeight*inputWidth*inputDepth);
    weights.imbue( [&]() { return getRandomValueBetweenBorders(-255, 255); } );

    biases = arma::zeros(numOutputNeurons);
    biases.randu();

    _resetAccumulatedGradients();
}

arma::cube& SoftmaxLayer::feedForward(arma::cube& input)
{
    this->input = input;
    auto* activationFunction = new SigmoidActivation();

    output = arma::zeros(numOutputNeurons,1,1);
    zWeightedInput = arma::zeros(numOutputNeurons);
    activationOutput = arma::zeros(numOutputNeurons);

    arma::vec vectorisedInput = arma::vectorise(input);
    zWeightedInput = (weights * vectorisedInput) + biases;
    activationOutput = activationFunction->forwardActivation(zWeightedInput);

    double sumExp = 0.0;
    for(size_t i = 0; i< vectorisedInput.size(); i++){
        sumExp += exp(vectorisedInput.at(i));
    }

    for(size_t i = 0; i< softmaxScores.size(); i++){
        softmaxScores(i) = exp(vectorisedInput.at(i))/sumExp;
        //softmaxLoss(i) = -log(softmaxScores(i));
    }

#if DEBUG
    std::cout << "\nSoftmax:" << std::endl;
    activationOutput.print();
#endif

    output.slice(0).col(0) = activationOutput;

    return output;
}

void SoftmaxLayer::backprop(arma::vec* upstreamGradient)
{
    if(upstreamGradient != nullptr){
        //Todo: Error!
    }

    deltaError = arma::zeros(numOutputNeurons);
    nablaWeights = arma::zeros(numOutputNeurons, inputHeight*inputWidth*inputDepth);
    nablaBiases = arma::zeros(numOutputNeurons);

    cost = costFunction->calculateCost(activationOutput, trainData->at(imageIndex)->getExpectedScore());
    arma::vec outputGradient = arma::zeros(activationOutput.size());
    outputGradient = costFunction->calculateCostGradient(activationOutput, trainData->at(imageIndex)->getExpectedScore());

    arma::vec activFunctDerivativesZ = arma::zeros(numOutputNeurons);
    auto* activationFunction = new SigmoidActivation();
    activFunctDerivativesZ = activationFunction->derivativeActivation(zWeightedInput);

    deltaError = outputGradient % activFunctDerivativesZ;

    nablaWeights = arma::zeros(numOutputNeurons, inputHeight*inputWidth*inputDepth);
    nablaWeights = deltaError * arma::vectorise(input).t();
    nablaBiases = deltaError;

    accumulatedNablaWeights += nablaWeights;
    accumulatedNablaBiases += nablaBiases;

    this->upstreamGradient = deltaError;

#if DEBUG
    std::cout << "\nbp_Softmax:" << std::endl;
    this->upstreamGradient.print();
#endif

    //this->upstreamGradient.print();
    //std::cout << "\n" << std::endl;
}

void SoftmaxLayer::init_for_testing(size_t inputHeight, size_t inputWidth, size_t inputDepth){
    this->inputHeight = inputHeight;
    this->inputWidth = inputWidth;
    this->inputDepth = inputDepth;

    outputHeight = 1;
    outputWidth = 1;
    outputDepth = numOutputNeurons;

    softmaxScores = arma::zeros(numOutputNeurons);

    weights = arma::zeros(numOutputNeurons, inputHeight*inputWidth*inputDepth);
    weights.imbue( [&]() { return getRandomValueBetweenBorders(-255, 255); } );

    biases = arma::zeros(numOutputNeurons);
    biases.randu();

    _resetAccumulatedGradients();
}

void SoftmaxLayer::setTrainData(std::vector<Image *> *trainData) {
    this->trainData = trainData;
}

void SoftmaxLayer::setImageIndex(size_t index) {
    this->imageIndex = index;
}

double SoftmaxLayer::getCost() const {
    return cost;
}

int SoftmaxLayer::getClassOfHighestScore() {
    int highestScoreIndex = 0;

    for(int i = 0; i < softmaxScores.size()-1; i++){
        if(softmaxScores(i+1) > softmaxScores(i)){
            highestScoreIndex = i+1;
        }
    }
    return highestScoreIndex;
}

vector<Image *> *SoftmaxLayer::getTrainData() const {
    return trainData;
}

size_t SoftmaxLayer::getImageIndex() const {
    return imageIndex;
}

void SoftmaxLayer::_resetAccumulatedGradients()
{
    accumulatedNablaWeights = arma::zeros(numOutputNeurons, inputHeight*inputWidth*inputDepth);
    accumulatedNablaBiases = arma::zeros(numOutputNeurons);
}

double SoftmaxLayer::getRandomValueBetweenBorders(int min, int max){
    return ((rand()%(max-min + 1) + min)/(double)(max-min));
}

void SoftmaxLayer::updateWeightsAndBiases(size_t batchSize, double learningRate)
{
    weights = weights - (learningRate/batchSize) * accumulatedNablaWeights;
    biases = biases - (learningRate/batchSize) * accumulatedNablaBiases;

    _resetAccumulatedGradients();
}

const arma::mat &SoftmaxLayer::getWeights() const {
    return weights;
}

void SoftmaxLayer::setWeights(const arma::mat &weights) {
    SoftmaxLayer::weights = weights;
}

void SoftmaxLayer::setBiases(const arma::vec &biases) {
    SoftmaxLayer::biases = biases;
}

const arma::vec &SoftmaxLayer::getZWeightedInput() const {
    return zWeightedInput;
}

const arma::vec &SoftmaxLayer::getDeltaError() const {
    return deltaError;
}

const arma::mat &SoftmaxLayer::getNablaWeights() const {
    return nablaWeights;
}

const arma::vec &SoftmaxLayer::getNablaBiases() const {
    return nablaBiases;
}

#undef DEBUG
