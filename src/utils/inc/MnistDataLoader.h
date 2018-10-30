//
// Created by felix on 24.10.18.
//

#ifndef CNN_GPU_MNISTDATALOADER_H
#define CNN_GPU_MNISTDATALOADER_H

#include <armadillo>
#include <iostream>
#include "Image.h"

using namespace std;

class MnistDataLoader {
private:
    int ReverseInt(int i);

public:
    std::vector<Image*> readMnistData(string pathToImageFile, string pathToLabelFile);

};


#endif //CNN_GPU_MNISTDATALOADER_H
