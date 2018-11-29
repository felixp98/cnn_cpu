//
// Created by fpreuschoff on 23.10.2018.
//

#include <net/layers/inc/ConvolutionalLayer.h>
#include <net/layers/inc/MaxPoolLayer.h>
#include <net/layers/inc/SoftmaxLayer.h>
#include <net/cost/inc/QuadraticCost.h>
#include <net/layers/inc/FullyConnectedLayer.h>
#include <net/activation/inc/SigmoidActivation.h>
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
    auto* softmax = new SoftmaxLayer(new QuadraticCost(), 5);

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

TEST(layer_tests, fullyConnected_Forward_Test) {
    auto* fc = new FullyConnectedLayer(new SigmoidActivation(), 2);
    fc->init_for_testing(3,1,1);

    arma::cube input(3,1,1);
    input(0,0,0) = 0.7;
    input(1,0,0) = 0.6;
    input(2,0,0) = 0.3;
    //input.print();

    arma::mat weights = {{0.8, 0.5, -0.2},
                         {-0.6,  0.7, 0.9}};

    //weights.print();

    arma::vec bias = {0.3, -0.2};
    //bias.print();

    fc->setWeights(weights);
    fc->setBiases(bias);

    fc->feedForward(input);
    arma::vec output = fc->getZWeightedInput();

    EXPECT_TRUE(std::abs(output(0) - 1.1) < 0.000001);
    EXPECT_TRUE(std::abs(output(1) - 0.07) < 0.000001);
}

TEST(layer_tests, fullyConnected_Big_Test) {
    auto* fc = new FullyConnectedLayer(new SigmoidActivation(), 3);
    auto* softmax = new SoftmaxLayer(new QuadraticCost(), 2);
    fc->setBeforeLayer(nullptr);
    fc->setAfterLayer(softmax);
    softmax->setBeforeLayer(fc);
    softmax->setAfterLayer(nullptr);

    arma::mat temp = {{0.7, 0.3},{0.6, 0.8}};
    arma::cube inputImage = arma::zeros(2,2,1);
    inputImage.slice(0) = temp;
    auto* image = new Image(inputImage);
    image->setNumClasses(2);
    image->setClassIndex(0);

    std::vector<Image*> images;
    images.push_back(image);

    softmax->setTrainData(&images);
    softmax->setImageIndex(0);

    fc->init_for_testing(2,2,1);
    softmax->init_for_testing(3,1,1);

    arma::mat weights1 = {{ 0.8,  0.5, -0.2, -0.7},
                          {-0.6,  0.7,  0.9,  0.3},
                          { 0.5, -0.1,  0.4,  0.6}};

    arma::vec bias1 = {0.3, -0.2, 0.1};

    arma::mat weights2 = {{0.7, -0.2, 0.8},
                          {0.1, -0.5, 0.4}};

    arma::vec bias2 = {0.1, -0.4};

    fc->setWeights(weights1);
    fc->setBiases(bias1);
    softmax->setWeights(weights2);
    softmax->setBiases(bias2);

    arma::vec output1 = arma::vectorise(fc->feedForward(inputImage));
    EXPECT_TRUE(std::abs(output1(0) - 0.631812418) < 0.001);
    EXPECT_TRUE(std::abs(output1(1) - 0.576885261) < 0.001);
    EXPECT_TRUE(std::abs(output1(2) - 0.729087922) < 0.001);

    arma::vec zWeightedOutput1 = fc->getZWeightedInput();
    EXPECT_TRUE(std::abs(zWeightedOutput1(0) - 0.54) < 0.001);
    EXPECT_TRUE(std::abs(zWeightedOutput1(1) - 0.31) < 0.001);
    EXPECT_TRUE(std::abs(zWeightedOutput1(2) - 0.99) < 0.001);

    arma::vec output2 = arma::vectorise(softmax->feedForward(softmax->getBeforeLayer()->getOutput()));
    EXPECT_TRUE(std::abs(output2(0) - 0.73305146) < 0.001);
    EXPECT_TRUE(std::abs(output2(1) - 0.41735862) < 0.001);

    arma::vec zWeightedOutput2 = softmax->getZWeightedInput();
    EXPECT_TRUE(std::abs(zWeightedOutput2(0) - 1.01016) < 0.001);
    EXPECT_TRUE(std::abs(zWeightedOutput2(1) - (-0.333626)) < 0.001);

    softmax->backprop(nullptr);
    arma::vec deltaError1 = softmax->getDeltaError();
    EXPECT_TRUE(std::abs(deltaError1(0) - (-0.05223836349)) < 0.001);
    EXPECT_TRUE(std::abs(deltaError1(1) - 0.1014892638) < 0.001);

    arma::mat nablaWeights1 = softmax->getNablaWeights();
    EXPECT_TRUE(std::abs(nablaWeights1(0,0) - (-0.033)) < 0.001);
    EXPECT_TRUE(std::abs(nablaWeights1(0,1) - (-0.0301)) < 0.001);
    EXPECT_TRUE(std::abs(nablaWeights1(0,2) - (-0.0381)) < 0.001);
    EXPECT_TRUE(std::abs(nablaWeights1(1,0) - 0.0641) < 0.001);
    EXPECT_TRUE(std::abs(nablaWeights1(1,1) - 0.0585) < 0.001);
    EXPECT_TRUE(std::abs(nablaWeights1(1,2) - 0.074) < 0.001);

    fc->backprop(&(fc->getAfterLayer()->getUpstreamGradient()));
    arma::vec deltaError2 = fc->getDeltaError();
    EXPECT_TRUE(std::abs(deltaError2(0) - ) < 0.0001);
    EXPECT_TRUE(std::abs(deltaError2(1) - ) < 0.0001);

}

TEST(layer_tests, fullyConnected_Backward_Test){
    auto* fc = new FullyConnectedLayer(new SigmoidActivation(), 10);
    fc->init_for_testing(5,5,3);

    arma::cube input(5, 5, 3, arma::fill::randn);
    arma::mat weights = fc->getWeights();
    arma::vec output;

    output = arma::vectorise(fc->feedForward(input));

    arma::vec upstreamGradient = arma::ones(size(output));

    fc->backprop(&upstreamGradient);

    arma::vec nablaInput = fc->getUpstreamGradient();
    arma::mat nablaWeights = fc->getNablaWeights();

    arma::vec approxNablaInput = arma::zeros(size(nablaInput));
    arma::mat approxNablaWeights = arma::zeros(size(weights));

    double disturbance = 0.5e-5;
    for (size_t i=0; i<input.n_elem; i++)
    {
        input[i] += disturbance;
        output = arma::vectorise(fc->feedForward(input));
        double l1 = arma::accu(output);
        input[i] -= 2.0*disturbance;
        output = arma::vectorise(fc->feedForward(input));
        double l2 = arma::accu(output);
        approxNablaInput[i] = (l1-l2)/(2.0*disturbance);
        input[i] += disturbance;
    }

    for(size_t i=0; i<nablaInput.n_elem; i++){
        std::cout << nablaInput[i] << "   " << approxNablaInput[i] << std::endl;
    }

    EXPECT_TRUE(arma::approx_equal(nablaInput, approxNablaInput, "absdiff", disturbance));

    for(size_t i=0; i<weights.n_elem; ++i){
        weights[i] += disturbance;
        fc->setWeights(weights);
        output = arma::vectorise(fc->feedForward(input));
        double l1 = arma::accu(output);
        weights[i] -= 2.0*disturbance;
        fc->setWeights(weights);
        output = arma::vectorise(fc->feedForward(input));
        double l2 = arma::accu(output);
        approxNablaWeights[i] = (l1-l2)/(2.0*disturbance);
        weights[i] += disturbance;
        fc->setWeights(weights);
    }

    nablaWeights.print();
    approxNablaWeights.print();

    EXPECT_TRUE(arma::approx_equal(nablaWeights, approxNablaWeights, "absdiff", disturbance));
}