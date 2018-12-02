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
    arma::vec temp = value;
    temp.transform( [](double val) { return val > 0? 1 : 0; } );
    return temp;
}

double ReluActivation::reluActivation(double v){
    return v > 0 ? v : 0.0;
}