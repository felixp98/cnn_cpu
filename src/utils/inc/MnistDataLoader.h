//
// Created by felix on 24.10.18.
//

#ifndef CNN_GPU_MNISTDATALOADER_H
#define CNN_GPU_MNISTDATALOADER_H

#include <armadillo>
#include <list>

class MnistDataLoader {
private:
    std::list<arma::cube> trainData;
    std::list<arma::cube> validationData;
    std::list<arma::cube> testData;

public:

};


#endif //CNN_GPU_MNISTDATALOADER_H
