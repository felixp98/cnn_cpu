//
// Created by felix on 28.10.18.
//

#ifndef CNN_GPU_NETWORK_H
#define CNN_GPU_NETWORK_H

#include <list>
#include <net/layers/inc/Layer.h>
#include "utils/inc/Image.h"

class Network{
private:
    std::list<Layer*> layers;
    std::vector<Image*> trainData;
    std::vector<Image*> validationData;
    std::vector<Image*> testData;

    bool initialized = false;
    double error;
    size_t rawInputHeight;
    size_t rawInputWidth;
    size_t rawInputDepth;

public:
    void add(Layer* layer);
    void setTrainData(std::vector<Image*> *trainData);
    void setValidationData(std::vector<Image*> *validationData);
    void setTestData(std::vector<Image*> *testData);

    void init();
    void train(size_t epochs);
    double testEpoch();

    double getError() const;

    size_t getRawInputHeight() const;

    void setRawInputHeight(size_t rawInputHeight);

    size_t getRawInputWidth() const;

    void setRawInputWidth(size_t rawInputWidth);

    size_t getRawInputDepth() const;

    void setRawInputDepth(size_t rawInputDepth);

};

#endif //CNN_GPU_NETWORK_H
