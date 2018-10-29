//
// Created by felix on 28.10.18.
//

#include "Network.h"

void Network::add(Layer* layer) {
    if(initialized) std::cout << "Error: Add layers before initializing the network!" << std::endl;
    layers.push_back(layer);

    if(layers.size() > 1){
        Layer* beforeLayer = *std::next(layers.begin(), layers.size() - 2);
        layer->setBeforeLayer(beforeLayer);
        beforeLayer->setAfterLayer(layer);
    }else{
        layer->setBeforeLayer(nullptr);
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

void Network::testEpoch() {

}

double Network::getError() const {
    return error;
}