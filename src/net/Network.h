//
// Created by felix on 28.10.18.
//

#ifndef CNN_GPU_NETWORK_H
#define CNN_GPU_NETWORK_H

#include <list>
#include <net/layers/inc/Layer.h>

class Network{
private:
    std::list<Layer*> layers;
    std::list<arma::cube> trainData;
    std::list<arma::cube> validationData;
    std::list<arma::cube> testData;

    bool initialized = false;
    double error;
    size_t rawInputHeight;
    size_t rawInputWidth;
    size_t rawInputDepth;

    //Todo: add train data
    //Todo: add test data

public:
    void add(Layer* layer);
    void setTrainData(std::list<arma::cube> *trainData);
    void setValidationData(std::list<arma::cube> *validationData);
    void setTestData(std::list<arma::cube> *testData);

    void init();
    void trainEpoch();
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
