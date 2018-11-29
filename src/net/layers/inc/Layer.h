//
// Created by felix on 28.10.18.
//

#ifndef CNN_GPU_LAYER_H
#define CNN_GPU_LAYER_H

#include <iostream>
#include <armadillo>

typedef enum {
    INPUT_LAYER, CONV_LAYER, POOLING_LAYER, FULLY_CONNECTED_LAYER, SOFTMAX
} LAYER_TYPE;

class Layer{
private:
    LAYER_TYPE type;

    Layer* beforeLayer = nullptr;
    Layer* afterLayer = nullptr;

protected:
    arma::cube input;
    arma::cube output;

    arma::vec upstreamGradient;

    size_t inputHeight;
    size_t inputWidth;
    size_t inputDepth;

    size_t outputHeight;
    size_t outputWidth;
    size_t outputDepth;

public:
    virtual void init() = 0;
    virtual arma::cube& feedForward(arma::cube& input) = 0;
    virtual void backprop(arma::vec* upstreamGradient) = 0;

    arma::cube &getInput() {
        return input;
    }

    void setInput(arma::cube &input) {
        Layer::input = input;
    }

    arma::cube &getOutput() {
        return output;
    }

    void setOutput(arma::cube &output) {
        Layer::output = output;
    }

    Layer *getBeforeLayer() {
        return beforeLayer;
    }

    void setBeforeLayer(Layer *beforeLayer) {
        Layer::beforeLayer = beforeLayer;
    }

    Layer *getAfterLayer() const {
        return afterLayer;
    }

    void setAfterLayer(Layer *afterLayer) {
        Layer::afterLayer = afterLayer;
    }

    size_t getInputHeight() {
        return inputHeight;
    }

    size_t getInputWidth() {
        return inputWidth;
    }

    size_t getInputDepth() {
        return inputDepth;
    }

    size_t getOutputHeight() {
        return outputHeight;
    }

    void setOutputHeight(size_t outputHeight) {
        Layer::outputHeight = outputHeight;
    }

    size_t getOutputWidth() {
        return outputWidth;
    }

    void setOutputWidth(size_t outputWidth) {
        Layer::outputWidth = outputWidth;
    }

    size_t getOutputDepth() {
        return outputDepth;
    }

    void setOutputDepth(size_t outputDepth) {
        Layer::outputDepth = outputDepth;
    }

    void setInputHeight(size_t inputHeight) {
        Layer::inputHeight = inputHeight;
    }

    void setInputWidth(size_t inputWidth) {
        Layer::inputWidth = inputWidth;
    }

    void setInputDepth(size_t inputDepth) {
        Layer::inputDepth = inputDepth;
    }

    LAYER_TYPE getType() const {
        return type;
    }

    void setType(LAYER_TYPE type) {
        Layer::type = type;
    }

     arma::vec &getUpstreamGradient() {
        return upstreamGradient;
    }


};


#endif //CNN_GPU_LAYER_H
