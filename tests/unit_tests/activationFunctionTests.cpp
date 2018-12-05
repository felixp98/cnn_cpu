
//TODO: Tests have to be rewritten because of refactored layer structure!!!


 /*
#include <net/activation/inc/SigmoidActivation.h>
#include "gtest/gtest.h"

TEST(activationFunctionTests, sigmoidTest) {
    auto* sigmoidFunction =  new SigmoidActivation();

    arma::vec inputValues = {0.23, -5, 1.99, 55.4, -3.33};

    arma::vec outputValuesForward = sigmoidFunction->forwardActivation(inputValues);
    arma::vec sollOutputValuesForward = {0.557247855, 0.006692851, 0.879743138, 1.0, 0.034556230};
    EXPECT_TRUE(std::abs(outputValuesForward[0] - sollOutputValuesForward[0]) < 0.000001);
    EXPECT_TRUE(std::abs(outputValuesForward[1] - sollOutputValuesForward[1]) < 0.000001);
    EXPECT_TRUE(std::abs(outputValuesForward[2] - sollOutputValuesForward[2]) < 0.000001);
    EXPECT_TRUE(std::abs(outputValuesForward[3] - sollOutputValuesForward[3]) < 0.000001);
    EXPECT_TRUE(std::abs(outputValuesForward[4] - sollOutputValuesForward[4]) < 0.000001);

    arma::vec outputValuesBackward = sigmoidFunction->derivativeActivation(inputValues);
    arma::vec sollOutputValuesBackward = {0.246722683, 0.006648057, 0.105795150, 0.0, 0.033362097};
    EXPECT_TRUE(std::abs(outputValuesBackward[0] - sollOutputValuesBackward[0]) < 0.000001);
    EXPECT_TRUE(std::abs(outputValuesBackward[1] - sollOutputValuesBackward[1]) < 0.000001);
    EXPECT_TRUE(std::abs(outputValuesBackward[2] - sollOutputValuesBackward[2]) < 0.000001);
    EXPECT_TRUE(std::abs(outputValuesBackward[3] - sollOutputValuesBackward[3]) < 0.000001);
    EXPECT_TRUE(std::abs(outputValuesBackward[4] - sollOutputValuesBackward[4]) < 0.000001);
}
*/