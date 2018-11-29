//
// Created by felix on 20.11.18.
//

#ifndef CNN_GPU_QUADRATICCOST_H
#define CNN_GPU_QUADRATICCOST_H

#include "CostFunction.h"
#include <math.h>

class QuadraticCost : public CostFunction{
public:
    double calculateCost(arma::vec &output, arma::vec &expectedOutput) override;

    arma::vec calculateCostGradient(arma::vec &output, arma::vec &expectedOutput) override;
};

#endif //CNN_GPU_QUADRATICCOST_H
