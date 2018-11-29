//
// Created by felix on 20.11.18.
//

#include "../inc/QuadraticCost.h"

double QuadraticCost::calculateCost(arma::vec &output, arma::vec &expectedOutput) {
    double sum = 0.0;
    //Todo: check if output and expected output have the same length
    for(size_t i = 0; i < output.size(); ++i){
        sum += pow(output(i)-expectedOutput(i), 2);
    }
    return sum*0.5;
}

arma::vec QuadraticCost::calculateCostGradient(arma::vec &output, arma::vec &expectedOutput) {
    return (output - expectedOutput);
}
