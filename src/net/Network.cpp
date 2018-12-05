//
// Created by felix on 01.12.18.
//

#include "Network.h"
#include "layers/InputLayer.h"
#include "layers/CrossEntropyLossLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/ConvolutionalLayer.h"

Network::Network(double learningRate, size_t batchSize) {
    auto* inputLayer = new InputLayer();

    this->LEARNING_RATE = learningRate;
    this->BATCH_SIZE = batchSize;

    layers.push_back(inputLayer);
    inputLayer->setBeforeLayer(nullptr);
}

void Network::add(Layer *newLayer) {
    if (initialized) std::cout << "Error: Add layers before initializing the network!" << std::endl;

    layers.push_back(newLayer);

    Layer *beforeLayer = layers.at(layers.size() - 2);
    beforeLayer->setAfterLayer(newLayer);

    newLayer->setBeforeLayer(beforeLayer);
    newLayer->setAfterLayer(nullptr);
}

void Network::init() {
    initialized = true;

    TRAIN_DATA_SIZE = trainData.size();
    VALIDATION_DATA_SIZE = validationData.size();
    TEST_DATA_SIZE = testData.size();
    NUM_BATCHES = TRAIN_DATA_SIZE / BATCH_SIZE;

    layers.at(0)->setInputHeight(trainData.at(0)->getImageData().n_rows);
    layers.at(0)->setInputWidth(trainData.at(0)->getImageData().n_cols);
    layers.at(0)->setInputDepth(trainData.at(0)->getImageData().n_slices);

    for (auto &layer : layers) {
        layer->init();
    }
    epochIdx = 0;

    std::cout << "\nNetwork initialized" << std::endl;
}

void Network::trainEpoch() {
    std::cout << "\nEpoch Nr." << ++epochIdx << std::endl;

    double sumLoss = 0.0;

    for (size_t batchIdx = 0; batchIdx < NUM_BATCHES; batchIdx++) {
        std::cout << "[" << batchIdx*BATCH_SIZE << "|" << trainData.size() << "]\r" << std::flush;

        // Generate a random batch.
        arma::vec batch(BATCH_SIZE, arma::fill::randu);
        batch *= (TRAIN_DATA_SIZE - 1);

        for (size_t i = 0; i < BATCH_SIZE; i++) {
            //FeedForward
            layers.at(0)->setInput(trainData[batch[i]]->getImageData());
            for(size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx){
                if(layerIdx == layers.size()-1) {layers.at(layerIdx)->setExpectedOutput(trainData[batch[i]]->getExpectedScore());}
                layers.at(layerIdx)->feedForward();
                if(layerIdx == layers.size()-1 && layers.at(layerIdx)->getType() == CROSS_ENTROPY_COST_LAYER) {
                    sumLoss += ((CrossEntropyLossLayer*)layers.at(layerIdx))->getLoss();
                }
            }

            //Backpropagation
            for(size_t layerIdx = layers.size()-1; layerIdx > 0; --layerIdx){
                layers.at(layerIdx)->backPropagate();
            }
        }

        //Update Weights and Biases
        for(size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx){
            if(layers.at(layerIdx)->getType() == FULLY_CONNECTED_LAYER){
                auto* fcLayer = (FullyConnectedLayer*)layers.at(layerIdx);
                fcLayer->UpdateWeightsAndBiases(BATCH_SIZE, LEARNING_RATE);
            }else if(layers.at(layerIdx)->getType() == CONV_LAYER){
                auto* convLayer = (ConvolutionalLayer*)layers.at(layerIdx);
                convLayer->UpdateFilterWeights(BATCH_SIZE, LEARNING_RATE);
            }
        }
    }

    std::cout << "Average loss: " << sumLoss / (BATCH_SIZE * NUM_BATCHES) << std::endl;

    double correctImages = 0.0;
    /*
    //Compute training accuracy
    for (size_t i = 0; i < TRAIN_DATA_SIZE; i++)
    {
        int predictedIndex = 0;

        //FeedForward
        layers.at(0)->setInput(trainData[i]->getImageData());
        for(size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx){
            if(layerIdx == layers.size()-1) {layers.at(layerIdx)->setExpectedOutput(trainData[i]->getExpectedScore());}
            layers.at(layerIdx)->feedForward();
            if(layerIdx == layers.size()-1 && layers.at(layerIdx)->getType() == CROSS_ENTROPY_COST_LAYER) {
                predictedIndex = ((CrossEntropyLossLayer*)layers.at(layerIdx))->getMaxIndex();
            }
        }

        if(trainData[i]->getExpectedScore().index_max() == predictedIndex){
            correctImages++;
        }
    }
    std::cout << "Training accuracy: " << correctImages/TRAIN_DATA_SIZE << std::endl;
*/

    //Compute validation accuracy
    correctImages = 0.0;
    for (size_t i = 0; i < VALIDATION_DATA_SIZE; i++)
    {
        int predictedIndex = 0;

        //FeedForward
        layers.at(0)->setInput(validationData[i]->getImageData());
        for(size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx){
            if(layerIdx == layers.size()-1) {layers.at(layerIdx)->setExpectedOutput(validationData[i]->getExpectedScore());}
            layers.at(layerIdx)->feedForward();
            if(layerIdx == layers.size()-1 && layers.at(layerIdx)->getType() == CROSS_ENTROPY_COST_LAYER) {
                predictedIndex = ((CrossEntropyLossLayer*)layers.at(layerIdx))->getMaxIndex();
            }
        }

        if(validationData[i]->getExpectedScore().index_max() == predictedIndex){
            correctImages++;
        }
    }
    std::cout << "Validation accuracy: " << correctImages/VALIDATION_DATA_SIZE << std::endl;
}

void Network::testEpoch() {
    //Compute test accuracy
    double correctImages = 0.0;
    for (size_t i = 0; i < TEST_DATA_SIZE; i++)
    {
        int predictedIndex = 0;

        //FeedForward
        layers.at(0)->setInput(testData[i]->getImageData());
        for(size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx){
            if(layerIdx == layers.size()-1) {layers.at(layerIdx)->setExpectedOutput(testData[i]->getExpectedScore());}
            layers.at(layerIdx)->feedForward();
            if(layerIdx == layers.size()-1 && layers.at(layerIdx)->getType() == CROSS_ENTROPY_COST_LAYER) {
                predictedIndex = ((CrossEntropyLossLayer*)layers.at(layerIdx))->getMaxIndex();
            }
        }

        if(testData[i]->getExpectedScore().index_max() == predictedIndex){
            correctImages++;
        }
    }
    std::cout << "Test accuracy: " << correctImages/TEST_DATA_SIZE << std::endl;
    error = 1-correctImages/TEST_DATA_SIZE;
}

void Network::setTrainData(std::vector<Image*> &trainData) {
    Network::trainData = trainData;
}

void Network::setValidationData(std::vector<Image*> &validationData) {
    Network::validationData = validationData;
}

void Network::setTestData(std::vector<Image*> &testData) {
    Network::testData = testData;
}

double Network::getError() const {
    return error;
}
