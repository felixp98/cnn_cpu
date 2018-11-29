//
// Created by felix on 15.11.18.
//

#ifndef CNN_GPU_RELUACTIVATION_H
#define CNN_GPU_RELUACTIVATION_H

#include "ActivationFunction.h"

class ReluActivation : public ActivationFunction {
private:
    double reluActivation(double v);

public:
    arma::vec forwardActivation(arma::vec& value) override;

    arma::vec derivativeActivation(arma::vec& value) override;
};

#endif //CNN_GPU_RELUACTIVATION_H
