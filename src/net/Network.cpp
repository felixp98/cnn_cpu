#include "Network.h"
#include "layers/InputLayer.h"
#include "layers/CrossEntropyLossLayer.h"
#include "layers/FullyConnectedLayer.h"
#include "layers/ConvolutionalLayer.h"
#include "layers/QuadraticLossLayer.h"
#include <chrono>

#define TIME_MEASURE true

#if TIME_MEASURE
using namespace std::chrono;
#endif

Network::Network(double learningRate, size_t batchSize) {
    auto* inputLayer = new InputLayer();

    this->LEARNING_RATE = learningRate;
    this->BATCH_SIZE = batchSize;

    layers.push_back(inputLayer);
    inputLayer->setBeforeLayer(nullptr);
}

void Network::add(Layer *newLayer) {
    if (initialized) std::cout << "Error: Add layers before initializing the network!" << std::endl;

    layers.push_back(newLayer);

    Layer *beforeLayer = layers.at(layers.size() - 2);
    beforeLayer->setAfterLayer(newLayer);

    newLayer->setBeforeLayer(beforeLayer);
    newLayer->setAfterLayer(nullptr);
}

void Network::init() {
    initialized = true;

    TRAIN_DATA_SIZE = trainData.size();
    VALIDATION_DATA_SIZE = validationData.size();
    TEST_DATA_SIZE = testData.size();
    NUM_BATCHES = TRAIN_DATA_SIZE / BATCH_SIZE;

    layers.at(0)->setInputHeight(trainData.at(0)->getImageData().n_rows);
    layers.at(0)->setInputWidth(trainData.at(0)->getImageData().n_cols);
    layers.at(0)->setInputDepth(trainData.at(0)->getImageData().n_slices);

    for (auto &layer : layers) {
        layer->init();
    }
    epochIdx = 0;

    std::cout << "\nNetwork initialized" << std::endl;
}

void Network::trainEpoch() {
    std::cout << "\nEpoch Nr." << ++epochIdx << std::endl;

    double sumLoss = 0.0;

    for (size_t batchIdx = 0; batchIdx < NUM_BATCHES; batchIdx++) {
        std::cout << "[" << batchIdx*BATCH_SIZE << "|" << trainData.size() << "]\r" << std::flush;

        // Generate a random batch.
        arma::vec batch(BATCH_SIZE, arma::fill::randu);
        batch *= (TRAIN_DATA_SIZE - 1);

        for (size_t i = 0; i < BATCH_SIZE; i++) {
#if TIME_MEASURE
            high_resolution_clock::time_point t_forward_start = high_resolution_clock::now();
#endif
            //FeedForward
            layers.at(0)->setInput(trainData[batch[i]]->getImageData());
            for(size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx){
                if(layerIdx == layers.size()-1) {layers.at(layerIdx)->setExpectedOutput(trainData[batch[i]]->getExpectedScore());}
                layers.at(layerIdx)->feedForward();
                if(layerIdx == layers.size()-1 && layers.at(layerIdx)->getType() == CROSS_ENTROPY_COST_LAYER) {
                    sumLoss += ((CrossEntropyLossLayer*)layers.at(layerIdx))->getLoss();
                }
                if(layerIdx == layers.size()-1 && layers.at(layerIdx)->getType() == QUADRATIC_LOSS_LAYER) {
                    sumLoss += ((QuadraticLossLayer*)layers.at(layerIdx))->getLoss();
                }
            }
#if TIME_MEASURE
            high_resolution_clock::time_point t_forward_stop = high_resolution_clock::now();
            sumForwardDurationsMicroseconds += duration_cast<microseconds>(t_forward_stop-t_forward_start).count();
#endif

#if TIME_MEASURE
            high_resolution_clock::time_point t_backprop_start = high_resolution_clock::now();
#endif
            //Backpropagation
            for(size_t layerIdx = layers.size()-1; layerIdx > 0; --layerIdx){
                layers.at(layerIdx)->backPropagate();
            }
#if TIME_MEASURE
            high_resolution_clock::time_point t_backprop_stop = high_resolution_clock::now();
            sumBackwardDurationsMicroseconds += duration_cast<microseconds>(t_backprop_stop-t_backprop_start).count();
#endif
        }

        //Update Weights and Biases
        for(size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx){
            if(layers.at(layerIdx)->getType() == FULLY_CONNECTED_LAYER){
                auto* fcLayer = (FullyConnectedLayer*)layers.at(layerIdx);
                fcLayer->updateWeightsAndBiases(BATCH_SIZE, LEARNING_RATE);
            }else if(layers.at(layerIdx)->getType() == CONV_LAYER){
                auto* convLayer = (ConvolutionalLayer*)layers.at(layerIdx);
                convLayer->updateFilterWeights(BATCH_SIZE, LEARNING_RATE);
            }
        }
    }

#if TIME_MEASURE
    averageForwardDuration = sumForwardDurationsMicroseconds / (NUM_BATCHES*BATCH_SIZE);
    averageBackwardDuration = sumBackwardDurationsMicroseconds / (NUM_BATCHES*BATCH_SIZE);
#endif

    std::cout << "Average loss: " << sumLoss / (BATCH_SIZE * NUM_BATCHES) << std::endl;
//    std::cout << "Init duration in microseconds: " << init_duration_microseconds << std::endl;
    std::cout << "Average forward duration in microseconds: " << averageForwardDuration << std::endl;
    std::cout << "Average backward duration in microseconds: " << averageBackwardDuration << std::endl;
    std::cout << "Total forward duration in microseconds: " << sumForwardDurationsMicroseconds << std::endl;
    std::cout << "Total backward duration in microseconds: " << sumBackwardDurationsMicroseconds << std::endl;

    for(size_t layerIdx = 0; layerIdx < layers.size(); layerIdx++){
        std::cout << "Layer #" << (layerIdx+1) << " " << layers[layerIdx]->getType() << " forward average: " << layers[layerIdx]->forwardDuration/TRAIN_DATA_SIZE << std::endl;
        std::cout << "Layer #" << (layerIdx+1) << " " << layers[layerIdx]->getType() << " backward average: " << layers[layerIdx]->backwardDuration/TRAIN_DATA_SIZE << std::endl;
    }

    double correctImages = 0.0;
    /*
    //Compute training accuracy
    for (size_t i = 0; i < TRAIN_DATA_SIZE; i++)
    {
        int predictedIndex = 0;

        //FeedForward
        layers.at(0)->setInput(trainData[i]->getImageData());
        for(size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx){
            if(layerIdx == layers.size()-1) {layers.at(layerIdx)->setExpectedOutput(trainData[i]->getExpectedScore());}
            layers.at(layerIdx)->feedForward();
            if(layerIdx == layers.size()-1 && layers.at(layerIdx)->getType() == CROSS_ENTROPY_COST_LAYER) {
                predictedIndex = ((CrossEntropyLossLayer*)layers.at(layerIdx))->getMaxIndex();
            }
        }

        if(trainData[i]->getExpectedScore().index_max() == predictedIndex){
            correctImages++;
        }
    }
    std::cout << "Training accuracy: " << correctImages/TRAIN_DATA_SIZE << std::endl;
*/

    //Compute validation accuracy
    correctImages = 0.0;
    for (size_t i = 0; i < VALIDATION_DATA_SIZE; i++)
    {
        int predictedIndex = 0;

        //FeedForward
        layers.at(0)->setInput(validationData[i]->getImageData());
        for(size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx){
            if(layerIdx == layers.size()-1) {layers.at(layerIdx)->setExpectedOutput(validationData[i]->getExpectedScore());}
            layers.at(layerIdx)->feedForward();
            if(layerIdx == layers.size()-1 && layers.at(layerIdx)->getType() == CROSS_ENTROPY_COST_LAYER) {
                predictedIndex = ((CrossEntropyLossLayer*)layers.at(layerIdx))->getMaxIndex();
            }
            if(layerIdx == layers.size()-1 && layers.at(layerIdx)->getType() == QUADRATIC_LOSS_LAYER) {
                predictedIndex = ((QuadraticLossLayer*)layers.at(layerIdx))->getMaxIndex();
            }
        }

        if(validationData[i]->getExpectedScore().index_max() == predictedIndex){
            correctImages++;
        }
    }
    std::cout << "Validation accuracy: " << correctImages/VALIDATION_DATA_SIZE << std::endl;
}

void Network::testEpoch() {
    //Compute test accuracy
    double correctImages = 0.0;
    for (size_t i = 0; i < TEST_DATA_SIZE; i++)
    {
        int predictedIndex = 0;

        //FeedForward
        layers.at(0)->setInput(testData[i]->getImageData());
        for(size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx){
            if(layerIdx == layers.size()-1) {layers.at(layerIdx)->setExpectedOutput(testData[i]->getExpectedScore());}
            layers.at(layerIdx)->feedForward();
            if(layerIdx == layers.size()-1 && layers.at(layerIdx)->getType() == CROSS_ENTROPY_COST_LAYER) {
                predictedIndex = ((CrossEntropyLossLayer*)layers.at(layerIdx))->getMaxIndex();
            }
            if(layerIdx == layers.size()-1 && layers.at(layerIdx)->getType() == QUADRATIC_LOSS_LAYER) {
                predictedIndex = ((QuadraticLossLayer*)layers.at(layerIdx))->getMaxIndex();
            }
        }

        if(testData[i]->getExpectedScore().index_max() == predictedIndex){
            correctImages++;
        }
    }
    std::cout << "Test accuracy: " << correctImages/TEST_DATA_SIZE << std::endl;
    error = 1-correctImages/TEST_DATA_SIZE;
}

void Network::setTrainData(std::vector<Image*> &trainData) {
    Network::trainData = trainData;
}

void Network::setValidationData(std::vector<Image*> &validationData) {
    Network::validationData = validationData;
}

void Network::setTestData(std::vector<Image*> &testData) {
    Network::testData = testData;
}

double Network::getError() const {
    return error;
}
