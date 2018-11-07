//
// Created by fpreuschoff on 23.10.2018.
//

#include <net/layers/inc/ConvolutionalLayer.h>
#include "gtest/gtest.h"

TEST(layer_tests, convolution_test) {
    //Conv layer with 1 filter, 3x3 filter size and stride 1
    auto * conv = new ConvolutionalLayer(1, 3, 1);

    arma::cube input = arma::zeros(5,5,1);
    input(0,0,0) = 2;
    input(0,1,0) = 1;
    input(0,2,0) = 0;
    input(0,3,0) = 2;
    input(0,4,0) = 0;
    input(1,0,0) = 2;
    input(1,1,0) = 0;
    input(1,2,0) = 2;
    input(1,3,0) = 2;
    input(1,4,0) = 0;
    input(2,0,0) = 0;
    input(2,1,0) = 0;
    input(2,2,0) = 2;
    input(2,3,0) = 1;
    input(2,4,0) = 2;
    input(3,0,0) = 0;
    input(3,1,0) = 1;
    input(3,2,0) = 1;
    input(3,3,0) = 1;
    input(3,4,0) = 0;
    input(4,0,0) = 0;
    input(4,1,0) = 1;
    input(4,2,0) = 1;
    input(4,3,0) = 2;
    input(4,4,0) = 0;

    arma::cube filter = arma::zeros(3,3,1);
    filter(0,0,0) = 0;
    filter(0,1,0) = 1;
    filter(0,2,0) = 1;
    filter(1,0,0) = 0;
    filter(1,1,0) = -1;
    filter(1,2,0) = -1;
    filter(2,0,0) = 1;
    filter(2,1,0) = 1;
    filter(2,2,0) = 1;

    std::vector<arma::cube> filters;
    filters.push_back(filter);

    conv->init_for_testing(5,5,1,filters);
    arma::cube output = conv->feedForward(input);

    EXPECT_EQ(1, output(0,0,0));
    EXPECT_EQ(1, output(0,1,0));
    EXPECT_EQ(5, output(0,2,0));
    EXPECT_EQ(2, output(1,0,0));
    EXPECT_EQ(4, output(1,1,0));
    EXPECT_EQ(1, output(1,2,0));
    EXPECT_EQ(2, output(2,0,0));
    EXPECT_EQ(5, output(2,1,0));
    EXPECT_EQ(5, output(2,2,0));
}