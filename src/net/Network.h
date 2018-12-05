//
// Created by felix on 01.12.18.
//

#ifndef CPP_CNN_NETWORK_H
#define CPP_CNN_NETWORK_H

#include "layers/Layer.h"
#include <iostream>
#include <armadillo>
#include <utils/Image.h>

class Network {
private:
    std::vector<Layer *> layers;

    std::vector<Image*> trainData;
    std::vector<Image*> validationData;
    std::vector<Image*> testData;

    bool initialized = false;

    size_t TRAIN_DATA_SIZE;
    size_t VALIDATION_DATA_SIZE;
    size_t TEST_DATA_SIZE;
    double LEARNING_RATE = 0.05;
    size_t BATCH_SIZE = 10;
    size_t NUM_BATCHES;

    size_t epochIdx = 0;

    double error = 0.0;

public:
    Network(double learningRate, size_t batchSize);

    void add(Layer *layer);

    void setTrainData(std::vector<Image*> &trainData);

    void setValidationData(std::vector<Image*> &validationData);

    void setTestData(std::vector<Image*> &testData);

    void init();

    void trainEpoch();

    void testEpoch();

    double getError() const;
};


#endif //CPP_CNN_NETWORK_H
