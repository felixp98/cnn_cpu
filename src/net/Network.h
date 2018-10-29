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
    bool initialized = false;
    double error;
    //Todo: add train data
    //Todo: add test data

public:
    void add(Layer* layer);
    void setTrainData();
    void setTestData();

    void init();
    void trainEpoch();
    void testEpoch();

    double getError() const;

};

#endif //CNN_GPU_NETWORK_H
