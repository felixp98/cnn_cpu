#include <net/Network.h>
#include <utils/MnistDataLoader.h>
#include <net/cost/inc/QuadraticCost.h>
#include <net/layers/CrossEntropyLossLayer.h>
#include <net/layers/ReluLayer.h>
#include <net/layers/SigmoidLayer.h>
#include <net/layers/QuadraticLossLayer.h>
#include "net/layers/ConvolutionalLayer.h"
#include "net/layers/FullyConnectedLayer.h"
#include "net/layers/SoftmaxLayer.h"
#include "net/layers/MaxPoolingLayer.h"
#include <chrono>

#define TIME_MEASURE true

using std::cout;
using namespace std::chrono;

int main()
{
#if TIME_MEASURE
    size_t init_duration_microseconds = 0;
    high_resolution_clock::time_point t_init_start = high_resolution_clock::now();
#endif

    cout << "-- CNN Reference Test on CPU --\n" << endl;
    cout << "Loading Image Data..." << flush;

    MnistDataLoader mdLoader("/home/felix/CLionProjects/cnn_cpu/data", 0.9);

    cout << "done\n" << endl;

    std::vector<Image*> trainData = mdLoader.getTrainData();
    std::vector<Image*> validationData = mdLoader.getValidationData();
    std::vector<Image*> testData = mdLoader.getTestData();

    cout << "[NETWORK CONFIGURATION]" << endl;
    cout << "Training data size: " << trainData.size() << endl;
    cout << "Validation data size: " << validationData.size() << endl;
    cout << "Test data size: " << testData.size() << "\n" << endl;


    auto *network = new Network(0.05, 10);
    network->setTrainData(trainData);
    network->setValidationData(validationData);
    network->setTestData(testData);

    //network->add(new ConvolutionalLayer(6, 5, 1));
    //network->add(new ReluLayer());
    //network->add(new MaxPoolingLayer(2, 2));
    //network->add(new ConvolutionalLayer(16, 5, 1));
    //network->add(new ReluLayer());
    //network->add(new MaxPoolingLayer(2, 2));
    network->add(new FullyConnectedLayer(50));
    network->add(new SigmoidLayer());
    network->add(new FullyConnectedLayer(10));
    network->add(new SigmoidLayer());
    network->add(new SoftmaxLayer(10));
    network->add(new CrossEntropyLossLayer(10));
    //network->add(new QuadraticLossLayer(10));

    network->init();

#if TIME_MEASURE
    high_resolution_clock::time_point t_init_stop = high_resolution_clock::now();
    init_duration_microseconds = duration_cast<milliseconds>(t_init_stop-t_init_start).count();
#endif

    std::cout << "Init duration microseconds: " << init_duration_microseconds << std::endl;

    do{
        network->trainEpoch();
        network->testEpoch();
    }while(network->getError() > 0.07);



    delete network;
}