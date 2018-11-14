#include "../inc/FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(size_t depth){
    std::cout << "new FullyConnectedLayer" << std::endl;
    this->depth = depth;
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
    weights.randu();

    biases = arma::zeros(depth);

    _resetAccumulatedGradients();
}

arma::cube& FullyConnectedLayer::feedForward(arma::cube& input)
{
    this->input = input;

    output = arma::zeros(depth,1,1);
    arma::vec vOutput = arma::zeros(depth);

    arma::vec vectorisedInput = arma::vectorise(input);
    vOutput = (weights * vectorisedInput) + biases;

    output.slice(0).col(0) = vOutput;

    output /= 100;


    return output;
}

void FullyConnectedLayer::backprop(arma::vec& upstreamGradient)
{
    std::cout << "backprop FC" << std::endl;

    arma::vec gradInputVec = arma::zeros(inputHeight*inputWidth*inputDepth);
    for (size_t i=0; i<(inputHeight*inputWidth*inputDepth); i++)
        gradInputVec[i] = arma::dot(weights.col(i), upstreamGradient);
    arma::cube tmp((inputHeight*inputWidth*inputDepth), 1, 1);
    tmp.slice(0).col(0) = gradInputVec;
    gradInput = arma::reshape(tmp, inputHeight, inputWidth, inputDepth);

    accumulatedGradInput += gradInput;

    gradWeights = arma::zeros(arma::size(weights));
    for (size_t i=0; i<gradWeights.n_rows; i++)
        gradWeights.row(i) = vectorise(input).t() * upstreamGradient[i];

    accumulatedGradWeights += gradWeights;

    gradBiases = upstreamGradient;
    accumulatedGradBiases += gradBiases;
}

void FullyConnectedLayer::_resetAccumulatedGradients()
{
    accumulatedGradInput = arma::zeros(inputHeight, inputWidth, inputDepth);
    accumulatedGradWeights = arma::zeros(
            depth,
            inputHeight*inputWidth*inputDepth
    );
    accumulatedGradBiases = arma::zeros(depth);
}

void FullyConnectedLayer::UpdateWeightsAndBiases(size_t batchSize, double learningRate)
{
    weights = weights - learningRate * (accumulatedGradWeights/batchSize);
    biases = biases - learningRate * (accumulatedGradBiases/batchSize);
    _resetAccumulatedGradients();
}