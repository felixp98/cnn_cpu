#ifndef CNN_GPU_CONVOLUTIONALLAYER_H
#define CNN_GPU_CONVOLUTIONALLAYER_H

#include "Layer.h"

class ConvolutionalLayer : public Layer {
private:
    size_t filterSize;
    size_t stride;
    size_t numFilters;

    std::vector<arma::cube> filters;

    double getRandomVal(double mean, double variance);

    void resetAccumulatedNablas();

    std::vector<arma::cube> nablaFilters;
    std::vector<arma::cube> accumulatedNablaFilters;

public:
    ConvolutionalLayer(size_t numFilters, size_t filterSize, size_t stride);

    void init() override;

    void feedForward() override;

    void backPropagate() override;

    void updateFilterWeights(size_t batchSize, double learningRate);

    void setFilters(std::vector<arma::cube> filters);

    std::vector<arma::cube> getFilters();

    std::vector<arma::cube> getNablaFilters();
};

#endif //CNN_GPU_CONVOLUTIONALLAYER_H
