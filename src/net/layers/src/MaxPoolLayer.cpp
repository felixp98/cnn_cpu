#include "../inc/MaxPoolLayer.h"

MaxPoolLayer::MaxPoolLayer(size_t poolingSize, size_t stride){
    std::cout << "new MaxPoolLayer" << std::endl;
    this->poolingSize = poolingSize;
    this->stride = stride;
}

void MaxPoolLayer::init()
{
    std::cout << "init maxpool" << std::endl;

    //init in-/outputsize
    if(getBeforeLayer() != nullptr){
        // Hidden Layer
        this->inputHeight = getBeforeLayer()->getOutputHeight();
        this->inputWidth = getBeforeLayer()->getOutputWidth();
        this->inputDepth = getBeforeLayer()->getOutputDepth();

        // if this is the first layer of the network, the input sizes got set directly when the layer was added
    }

    outputHeight = (inputHeight - poolingSize)/stride + 1;
    outputWidth = (inputWidth - poolingSize)/stride + 1;
    outputDepth = inputDepth;
}

arma::cube& MaxPoolLayer::feedForward(arma::cube& input)
{
    //set input if layer is hidden-layer
    if(getBeforeLayer() != nullptr){
        input = getBeforeLayer()->getOutput();
    }

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

void MaxPoolLayer::backprop()
{
    std::cout << "backprop maxpool" << std::endl;
}