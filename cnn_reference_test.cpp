#include <net/Network.h>
#include <utils/inc/MnistDataLoader.h>
#include "net/layers/inc/ConvolutionalLayer.h"
#include "net/layers/inc/FullyConnectedLayer.h"
#include "net/layers/inc/SoftmaxLayer.h"
#include "net/layers/inc/MaxPoolLayer.h"
#include "net/layers/inc/ReluLayer.h"

using std::cout;

int main()
{
	cout << "CNN Reference Test CPU:" << endl;

    cout << "Loading Training Image Data..." << std::flush;
    auto * dataLoader = new MnistDataLoader();
    std::vector<Image*> trainData = dataLoader->readMnistData("/home/felix/MNIST/train-images-idx3-ubyte", "/home/felix/MNIST/train-labels-idx1-ubyte");
    cout << " done" << endl;
    std::vector<Image*> validationData;
    cout << "Loading Test Image Data..." << std::flush;
    std::vector<Image*> testData = dataLoader->readMnistData("/home/felix/MNIST/t10k-images-idx3-ubyte", "/home/felix/MNIST/t10k-labels-idx1-ubyte");
    cout << " done" << endl;

    //cout << "Label: " << trainData.at(0)->getLabel() << endl;
    //trainData.at(0)->getImageData().print();
    //cout << "Label: " << testData.at(0)->getLabel() << endl;
    //testData.at(0)->getImageData().print();

    //create Network structure
    auto * network = new Network();

    network->setTrainData(&trainData);
    network->setValidationData(&validationData);
    network->setTestData(&testData);

    //network->add(new ConvolutionalLayer(10, 5, 1));
    //network->add(new MaxPoolLayer(2, 2));
    //network->add(new ReluLayer());
    network->add(new FullyConnectedLayer(1000));
    network->add(new ReluLayer());
    network->add(new FullyConnectedLayer(10));
    network->add(new ReluLayer());
    network->add(new SoftmaxLayer(10));

    network->init();


    do {
        network->trainEpoch();
    }while (network->testEpoch() > 10.0);


    delete network;

	return 0;
}