//
// Created by felix on 20.11.18.
//

#include <armadillo>
#include "../inc/CrossEntropyCost.h"
#include <math.h>

double CrossEntropyCost::calculateCost(arma::vec &output, arma::vec &expectedOutput) {
    double sum = 0.0;
    //Todo: check if output and expected output have the same length
    for(size_t i = 0; i < output.size(); ++i){
        sum += expectedOutput(i) * log(output(i))
                + (1-expectedOutput(i)) * log(1-output(i));
    }
    return -(sum/output.size());
}
