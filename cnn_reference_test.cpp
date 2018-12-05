#include <net/Network.h>
#include <utils/MnistDataLoader.h>
#include <net/cost/inc/QuadraticCost.h>
#include <net/layers/CrossEntropyLossLayer.h>
#include <net/layers/ReluLayer.h>
#include <net/layers/SigmoidLayer.h>
#include "net/layers/ConvolutionalLayer.h"
#include "net/layers/FullyConnectedLayer.h"
#include "net/layers/SoftmaxLayer.h"
#include "net/layers/MaxPoolingLayer.h"

using std::cout;

int main()
{
	cout << "-- CNN Reference Test on CPU --\n" << endl;
    cout << "Loading Image Data..." << flush;

    MnistDataLoader mdLoader("/home/felix/CLionProjects/cnn_cpu/data");

    cout << "done\n" << endl;

    std::vector<Image*> trainData = mdLoader.getTrainData();
    std::vector<Image*> validationData = mdLoader.getValidationData();
    std::vector<Image*> testData = mdLoader.getTestData();

    cout << "[NETWORK CONFIGURATION]" << endl;
    cout << "Training data size: " << trainData.size() << endl;
    cout << "Validation data size: " << validationData.size() << endl;
    cout << "Test data size: " << testData.size() << "\n" << endl;


    auto *network = new Network(0.05, 100);
    network->setTrainData(trainData);
    network->setValidationData(validationData);
    network->setTestData(testData);

    //network->add(new ConvolutionalLayer(6, 5, 1));
    //network->add(new ReluLayer());
    //network->add(new MaxPoolingLayer(2, 2));
    //network->add(new ConvolutionalLayer(16, 5, 1));
    //network->add(new ReluLayer());
    //network->add(new MaxPoolingLayer(2, 2));
    network->add(new FullyConnectedLayer(30));
    network->add(new SigmoidLayer());
    network->add(new FullyConnectedLayer(10));
    network->add(new SigmoidLayer());
    network->add(new SoftmaxLayer(10));
    network->add(new CrossEntropyLossLayer(10));

    network->init();

    network->trainEpoch();
    network->trainEpoch();
    network->trainEpoch();

    delete network;
}