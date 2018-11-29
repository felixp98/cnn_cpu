//
// Created by felix on 28.10.18.
//

#include <utils/inc/Image.h>
#include <net/layers/inc/SoftmaxLayer.h>
#include <net/layers/inc/FullyConnectedLayer.h>
#include "Network.h"


void Network::add(Layer* layer) {
    if(initialized) std::cout << "Error: Add layers before initializing the network!" << std::endl;
    layers.push_back(layer);
    //layer->setNetwork(this);

    if(layers.size() > 1){
        Layer* beforeLayer = *std::next(layers.begin(), layers.size() - 2);
        layer->setBeforeLayer(beforeLayer);
        beforeLayer->setAfterLayer(layer);
    }else{
        layer->setBeforeLayer(nullptr);
        layer->setInputHeight(trainData.at(0)->getImageData().n_rows);
        layer->setInputWidth(trainData.at(0)->getImageData().n_cols);
        layer->setInputDepth(trainData.at(0)->getImageData().n_slices);
    }
    layer->setAfterLayer(nullptr);
}

void Network::setTrainData(std::vector<Image*> *trainData) {
    this->trainData = *trainData;
}

void Network::setValidationData(std::vector<Image*> *validationData) {
    this->validationData = *validationData;
}

void Network::setTestData(std::vector<Image*> *testData) {
    this->testData = *testData;
}

void Network::init() {
    initialized = true;

    for(auto& layer : layers){
        layer->init();
        if(layer->getAfterLayer() == nullptr){
            ((SoftmaxLayer*)layer)->setTrainData(&(this->trainData));
        }
    }

    std::cout << "Network initialized" << std::endl;
}

void Network::train(size_t numEpochs) {
    double cost_sum_minibatch = 0.0;

    const size_t TRAIN_DATA_SIZE = trainData.size();
    const size_t VALIDATION_DATA_SIZE = validationData.size();
    const size_t TEST_DATA_SIZE = testData.size();
    const double LEARNING_RATE = 0.05;
    const size_t EPOCHS = numEpochs;
    const size_t BATCH_SIZE = 10;
    const size_t NUM_BATCHES = TRAIN_DATA_SIZE / BATCH_SIZE;


    for (size_t epoch = 0; epoch < EPOCHS; epoch++) {
        for (size_t batchIndex = 0; batchIndex < NUM_BATCHES; batchIndex++) {

            // Generate a random batch.
            arma::vec batch(BATCH_SIZE, arma::fill::randu);
            batch *= (TRAIN_DATA_SIZE - 1);

            for (size_t i = 0; i < BATCH_SIZE; i++) {
                for (auto &layer : layers) {
                    if (layer->getBeforeLayer() == nullptr) {
                        layer->feedForward(trainData.at(batch[i])->getImageData());
                    } else {
                        layer->feedForward(layer->getBeforeLayer()->getOutput());
                    }
                }
                for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
                    Layer *&currentLayer = *it;

                    if (currentLayer->getAfterLayer() == nullptr && currentLayer->getType() == SOFTMAX) {
                        auto *softmaxLayer = (SoftmaxLayer *) currentLayer;
                        softmaxLayer->setImageIndex(batch[i]);
                        softmaxLayer->backprop(nullptr);
                        cost_sum_minibatch += softmaxLayer->getCost();
                    } else {
                        currentLayer->backprop(&(currentLayer->getAfterLayer()->getUpstreamGradient()));
                    }
                }
                //break; //Todo: remove on target
            }
            //break;
            std::cout << cost_sum_minibatch << std::endl;
            cost_sum_minibatch = 0.0;
            for (auto &layer : layers) {
                if (layer->getType() == FULLY_CONNECTED_LAYER) {
                    ((FullyConnectedLayer *) layer)->updateWeightsAndBiases(BATCH_SIZE, LEARNING_RATE);
                }else if(layer->getType() == SOFTMAX){
                    ((SoftmaxLayer *) layer)->updateWeightsAndBiases(BATCH_SIZE, LEARNING_RATE);
                }
            }
            //break; //Todo: remove on target
        }
    }
}

double Network::testEpoch() {
    int correctClassCounter = 0;

    for(size_t imgIndex = 0; imgIndex < testData.size(); imgIndex++) {
        for (auto &layer : layers) {
            if (layer->getBeforeLayer() == nullptr) {
                layer->feedForward(testData.at(imgIndex)->getImageData());
            } else {
                layer->feedForward(layer->getBeforeLayer()->getOutput());
            }

            if(layer->getAfterLayer() == nullptr){
                int classIndex = ((SoftmaxLayer*)layer)->getClassOfHighestScore();
                if(classIndex == testData.at(imgIndex)->getClassIndex()){
                    correctClassCounter++;
                }
            }
        }
    }

    return (double)correctClassCounter/testData.size();
}

double Network::getError() const {
    return error;
}

size_t Network::getRawInputHeight() const {
    return rawInputHeight;
}

void Network::setRawInputHeight(size_t rawInputHeight) {
    Network::rawInputHeight = rawInputHeight;
}

size_t Network::getRawInputWidth() const {
    return rawInputWidth;
}

void Network::setRawInputWidth(size_t rawInputWidth) {
    Network::rawInputWidth = rawInputWidth;
}

size_t Network::getRawInputDepth() const {
    return rawInputDepth;
}

void Network::setRawInputDepth(size_t rawInputDepth) {
    Network::rawInputDepth = rawInputDepth;
}
