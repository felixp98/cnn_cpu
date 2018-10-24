#include "net/layers/inc/ConvolutionalLayer.h"
#include "net/layers/inc/FullyConnectedLayer.h"
#include "net/layers/inc/SoftmaxLayer.h"
#include "net/layers/inc/MaxPoolLayer.h"
#include "net/layers/inc/ReluLayer.h"

using std::cout;

int main()
{
	cout << "CNN Reference Test CPU:" << std::endl;

    auto* conv = new ConvolutionalLayer(5,5);
    auto* pool = new MaxPoolLayer();
    auto* relu = new ReluLayer();
    auto* fullyConnected = new FullyConnectedLayer();
    auto* softmax = new SoftmaxLayer();

    delete conv;
    delete pool;
    delete relu;
    delete fullyConnected;
    delete softmax;

	return 0;
}