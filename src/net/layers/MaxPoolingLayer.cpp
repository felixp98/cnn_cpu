#include "MaxPoolingLayer.h"

MaxPoolingLayer::MaxPoolingLayer(size_t poolingSize, size_t stride)  :
        poolingSize(poolingSize),
        stride(stride) {
    setType(POOLING_LAYER);
}

void MaxPoolingLayer::init() {
    this->inputHeight = getBeforeLayer()->getOutputHeight();
    this->inputWidth = getBeforeLayer()->getOutputWidth();
    this->inputDepth = getBeforeLayer()->getOutputDepth();

    outputHeight = (inputHeight - poolingSize)/stride + 1;
    outputWidth = (inputWidth - poolingSize)/stride + 1;
    outputDepth = inputDepth;

    //std::cout << "Input Pool: " << inputHeight << "x" << inputWidth << "x" << inputDepth << std::endl;
    std::cout << "Output Pool: " << outputHeight << "x" << outputWidth << "x" << outputDepth << std::endl;
}

void MaxPoolingLayer::feedForward() {
    this->input = getBeforeLayer()->getOutput();

    output = arma::zeros(outputHeight, outputWidth, outputDepth);
    for (size_t sidx = 0; sidx < inputDepth; sidx ++) {
        for (size_t ridx = 0; ridx <= inputHeight - poolingSize; ridx += stride) {
            for (size_t cidx = 0; cidx <= inputWidth - poolingSize; cidx += stride) {
                output.slice(sidx)(ridx / stride, cidx / stride) =
                        input.slice(sidx).submat(ridx, cidx, ridx + poolingSize - 1, cidx + poolingSize - 1).max();
            }
        }
    }
}

void MaxPoolingLayer::backPropagate() {
    arma::cube upstreamGradient = getAfterLayer()->getGradientInput();

    gradientInput = arma::zeros(inputHeight, inputWidth, inputDepth);
    for (size_t i = 0; i < inputDepth; i++) {
        for (size_t r = 0;
             r + poolingSize <= inputHeight;
             r += stride) {
            for (size_t c = 0;
                 c + poolingSize <= inputWidth;
                 c += stride) {
                arma::mat tmp(poolingSize,
                              poolingSize,
                              arma::fill::zeros);
                tmp(input.slice(i).submat(r, c,
                                          r + poolingSize - 1, c + poolingSize - 1)
                            .index_max()) = upstreamGradient.slice(i)(r / stride,
                                                                      c / stride);
                gradientInput.slice(i).submat(r, c,
                                              r + poolingSize - 1, c + poolingSize - 1) += tmp;
            }
        }
    }
}

