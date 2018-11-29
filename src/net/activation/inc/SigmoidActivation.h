//
// Created by felix on 15.11.18.
//

#ifndef CNN_GPU_SIGMOIDACTIVATION_H
#define CNN_GPU_SIGMOIDACTIVATION_H

#include "ActivationFunction.h"
#include <armadillo>

class SigmoidActivation : public ActivationFunction{
private:

public:
    arma::vec forwardActivation(arma::vec& value) override;

    arma::vec derivativeActivation(arma::vec& value) override;
};

#endif //CNN_GPU_SIGMOIDACTIVATION_H
