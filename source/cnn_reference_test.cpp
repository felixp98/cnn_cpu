#include "layers/inc/ConvolutionalLayer.h"

using namespace std;

int main()
{
	cout << "CNN Reference Test CPU:" << endl;

    ConvolutionalLayer* conv = new ConvolutionalLayer(5,5);
    delete conv;

	return 0;
}
