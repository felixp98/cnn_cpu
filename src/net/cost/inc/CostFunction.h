//
// Created by felix on 20.11.18.
//

#ifndef CNN_GPU_COSTFUNCTION_H
#define CNN_GPU_COSTFUNCTION_H

#include <armadillo>

class CostFunction{
public:
    virtual double calculateCost(arma::vec& output, arma::vec& expectedOutput) = 0;
    virtual arma::vec calculateCostGradient(arma::vec& output, arma::vec& expectedOutput) = 0;
};

#endif //CNN_GPU_COSTFUNCTION_H
