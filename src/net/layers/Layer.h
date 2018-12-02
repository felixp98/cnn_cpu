#ifndef CNN_GPU_LAYER_H
#define CNN_GPU_LAYER_H

#include <iostream>
#include <armadillo>

typedef enum {
    INPUT_LAYER, CONV_LAYER, POOLING_LAYER, FULLY_CONNECTED_LAYER, SOFTMAX_LAYER, CROSS_ENTROPY_COST_LAYER, RELU_LAYER,
    SIGMOID_LAYER
} LAYER_TYPE;

class Layer {
private:
    LAYER_TYPE type;

    Layer *beforeLayer = nullptr;
    Layer *afterLayer = nullptr;

protected:
    arma::cube input;
    arma::cube output;
    arma::cube gradientInput;

    size_t inputHeight;
    size_t inputWidth;
    size_t inputDepth;

    size_t outputHeight;
    size_t outputWidth;
    size_t outputDepth;

    arma::vec expectedOutput;

public:
    virtual void init() = 0;
    virtual void feedForward() = 0;
    virtual void backPropagate() = 0;

    LAYER_TYPE getType() const {
        return type;
    }

    void setType(LAYER_TYPE type) {
        Layer::type = type;
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

    void setInput(const arma::cube &input) {
        Layer::input = input;
    }

    const arma::cube &getInput() const {
        return input;
    }

    const arma::cube &getOutput() const {
        return output;
    }

    const arma::cube &getGradientInput() const {
        return gradientInput;
    }

    size_t getInputHeight() const {
        return inputHeight;
    }

    void setInputHeight(size_t inputHeight) {
        Layer::inputHeight = inputHeight;
    }

    size_t getInputWidth() const {
        return inputWidth;
    }

    void setInputWidth(size_t inputWidth) {
        Layer::inputWidth = inputWidth;
    }

    size_t getInputDepth() const {
        return inputDepth;
    }

    void setInputDepth(size_t inputDepth) {
        Layer::inputDepth = inputDepth;
    }

    size_t getOutputHeight() const {
        return outputHeight;
    }

    void setOutputHeight(size_t outputHeight) {
        Layer::outputHeight = outputHeight;
    }

    size_t getOutputWidth() const {
        return outputWidth;
    }

    void setOutputWidth(size_t outputWidth) {
        Layer::outputWidth = outputWidth;
    }

    size_t getOutputDepth() const {
        return outputDepth;
    }

    void setOutputDepth(size_t outputDepth) {
        Layer::outputDepth = outputDepth;
    }

    void setExpectedOutput(const arma::vec &expectedOutput) {
        Layer::expectedOutput = expectedOutput;
    }
};

#endif //CNN_GPU_LAYER_H
