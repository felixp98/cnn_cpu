//
// Created by felix on 28.10.18.
//

#include <utils/inc/Image.h>
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
    }

    std::cout << "Network initialized" << std::endl;
}

void Network::trainEpoch() {
    std::cout << "\r[1|" << trainData.size() << "] - Feedforward";
    for (int i = 0; i < trainData.size(); i++) {
        if((i+1)%100 == 0){
            std::cout << "\r[" << i+1 << "|" << trainData.size() << "] - Feedforward";
            std::cout.flush();
        }
        for (auto &layer : layers) {
            if(layer->getBeforeLayer() == nullptr){
                layer->setInput(trainData.at(i)->getImageData());
            }
            layer->feedForward();
        }
        break; //Todo: remove on target
    }

    //TODO: implement backpropagation
    /*for(auto& layer : layers){
        layer->backprop();
    }*/
}

double Network::testEpoch() {
    return 0.0;
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
