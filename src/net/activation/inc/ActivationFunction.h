//
// Created by felix on 15.11.18.
//

#ifndef CNN_GPU_ACTIVATIONFUNCTION_H
#define CNN_GPU_ACTIVATIONFUNCTION_H

#include <armadillo>

class ActivationFunction{
public:
    virtual arma::vec forwardActivation(arma::vec& value) = 0;
    virtual arma::vec derivativeActivation(arma::vec& value) = 0;
};

#endif //CNN_GPU_ACTIVATIONFUNCTION_H
