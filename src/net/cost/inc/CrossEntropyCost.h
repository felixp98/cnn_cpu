//
// Created by felix on 20.11.18.
//

#ifndef CNN_GPU_CROSSENTROPYCOST_H
#define CNN_GPU_CROSSENTROPYCOST_H

#include "CostFunction.h"

class CrossEntropyCost : public CostFunction{
public:
    double calculateCost(arma::vec &output, arma::vec &expectedOutput) override;
};

#endif //CNN_GPU_CROSSENTROPYCOST_H
