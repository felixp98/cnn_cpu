//
// Created by fpreuschoff on 23.10.2018.
//

#include <net/layers/inc/ConvolutionalLayer.h>
#include <net/layers/inc/MaxPoolLayer.h>
#include <net/layers/inc/SoftmaxLayer.h>
#include "gtest/gtest.h"

TEST(layer_tests, convolution_test) {
    //Conv layer with 1 filter, 3x3 filter size and stride 1
    auto * conv = new ConvolutionalLayer(1, 3, 1);

    arma::cube input = arma::zeros(5,5,1);
    input.slice(0) = {{2,1,0,2,0},
                      {2,0,2,2,0},
                      {0,0,2,1,2},
                      {0,1,1,1,0},
                      {0,1,1,2,0}};

    arma::cube filter1 = arma::zeros(3,3,1);
    filter1.slice(0) = {{0, 1, 1},
                       {0,-1,-1},
                       {1, 1, 1}};

    std::vector<arma::cube> filters;
    filters.push_back(filter1);

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

TEST(layer_tests, pooling_test){
    auto* maxPooling = new MaxPoolLayer(2, 2);

    arma::cube input = arma::zeros(4,4,2);
    input.slice(0) = {{5,2,7,1},
                      {2,3,0,8},
                      {3,1,1,4},
                      {4,4,7,2}};
    input.slice(1) = {{3,0,5,1},
                      {9,5,2,4},
                      {1,1,8,0},
                      {0,2,4,1}};

    maxPooling->init_for_testing(4,4,2);
    arma::cube output = maxPooling->feedForward(input);

    EXPECT_EQ(5, output(0,0,0));
    EXPECT_EQ(8, output(0,1,0));
    EXPECT_EQ(4, output(1,0,0));
    EXPECT_EQ(7, output(1,1,0));
    EXPECT_EQ(9, output(0,0,1));
    EXPECT_EQ(5, output(0,1,1));
    EXPECT_EQ(2, output(1,0,1));
    EXPECT_EQ(8, output(1,1,1));
}

TEST(layer_tests, softmax_test){
    auto* softmax = new SoftmaxLayer(5);

    arma::cube input = arma::zeros(5,1,1);
    arma::vec vInput = {5,1,3,9,7};
    input.slice(0).col(0) = vInput;

    softmax->init_for_testing(5,1,1);
    arma::cube output = softmax->feedForward(input);

    EXPECT_EQ("0.015838", std::to_string(output(0,0,0)));
    EXPECT_EQ("0.000290", std::to_string(output(1,0,0)));
    EXPECT_EQ("0.002143", std::to_string(output(2,0,0)));
    EXPECT_EQ("0.864704", std::to_string(output(3,0,0)));
    EXPECT_EQ("0.117025", std::to_string(output(4,0,0)));
}