//
// Created by felix on 15.11.18.
//

#include "../inc/ReluActivation.h"

arma::vec ReluActivation::forwardActivation(arma::vec& value) {
    //replace negative values with 0

    arma::vec temp = arma::zeros(value.size());
    for(int i = 0; i < value.size(); i++){
        temp(i) = reluActivation(value(i));
    }

    return temp;
}

arma::vec ReluActivation::derivativeActivation(arma::vec& value) {
    //TODO not implemented yet
    return arma::zeros(value.size());
}

double ReluActivation::reluActivation(double v){
    return v > 0 ? v : 0.0;
}