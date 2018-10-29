//
// Created by felix on 28.10.18.
//

#ifndef CNN_GPU_LAYER_H
#define CNN_GPU_LAYER_H

#include <iostream>
#include <armadillo>

class Layer{
private:
    Layer* beforeLayer = nullptr;
    Layer* afterLayer = nullptr;

protected:
    size_t depth;
    arma::cube input;
    arma::cube output;

    size_t inputHeight;
    size_t inputWidth;
    size_t inputDepth;

    size_t outputHeight;
    size_t outputWidth;
    size_t outputDepth;

public:

    virtual void init() = 0;
    virtual void feedForward() = 0;
    virtual void backprop() = 0;

    const arma::cube &getInput() const {
        return input;
    }

    void setInput(const arma::cube &input) {
        Layer::input = input;
    }

    const arma::cube &getOutput() const {
        return output;
    }

    void setOutput(const arma::cube &output) {
        Layer::output = output;
    }

    Layer *getBeforeLayer() const {
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

    size_t getDepth() const {
        return depth;
    }

    void setDepth(size_t depth) {
        Layer::depth = depth;
    }

    size_t getInputHeight() const {
        return inputHeight;
    }

    size_t getInputWidth() const {
        return inputWidth;
    }

    size_t getInputDepth() const {
        return inputDepth;
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
};


#endif //CNN_GPU_LAYER_H
