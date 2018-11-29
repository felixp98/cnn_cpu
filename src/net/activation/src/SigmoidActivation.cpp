//
// Created by felix on 15.11.18.
//

#include "../inc/SigmoidActivation.h"
#include <math.h>
#include <armadillo>

arma::vec SigmoidActivation::forwardActivation(arma::vec& value) {
    //return (1.0/(1.0 + exp(-value)));
    return (1.0/(1.0 + arma::exp(-value)));
}

arma::vec SigmoidActivation::derivativeActivation(arma::vec& value) {
    return forwardActivation(value) % (1.0 - forwardActivation(value));
}
