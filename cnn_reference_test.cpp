#include <net/Network.h>
#include "net/layers/inc/ConvolutionalLayer.h"
#include "net/layers/inc/FullyConnectedLayer.h"
#include "net/layers/inc/SoftmaxLayer.h"
#include "net/layers/inc/MaxPoolLayer.h"
#include "net/layers/inc/ReluLayer.h"

using std::cout;

int main()
{
	cout << "CNN Reference Test CPU:" << std::endl;

    auto* network = new Network();
    network->add(new ConvolutionalLayer(5, 5, 1));
    network->add(new MaxPoolLayer());
    network->add(new ReluLayer());
    network->add(new FullyConnectedLayer());
    network->add(new SoftmaxLayer());
    network->init();


    do {
        network->trainEpoch();
    }while (network->testEpoch() > 10.0);


    delete network;

	return 0;
}