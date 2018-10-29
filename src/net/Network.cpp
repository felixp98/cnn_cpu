//
// Created by felix on 28.10.18.
//

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
        layer->setInputHeight(/*rawDataHeight*/ 32);
        layer->setInputWidth(/*rawDataHeight*/ 32);
        layer->setInputDepth(/*rawDataHeight*/ 3);
    }
    layer->setAfterLayer(nullptr);
}

void Network::setTrainData() {
    //Todo
}

void Network::setTestData() {
    //Todo
}

void Network::init() {
    initialized = true;

    for(auto& layer : layers){
        layer->init();
    }
}

void Network::trainEpoch() {

}

double Network::testEpoch() {

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
