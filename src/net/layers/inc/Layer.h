//
// Created by felix on 28.10.18.
//

#ifndef CNN_GPU_LAYER_H
#define CNN_GPU_LAYER_H

#include <iostream>

class Layer{
private:
    Layer* beforeLayer = nullptr;
    Layer* afterLayer = nullptr;
    size_t depth;

public:

    virtual void init() = 0;
    virtual void feedForward() = 0;
    virtual void backprop() = 0;

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
};


#endif //CNN_GPU_LAYER_H
