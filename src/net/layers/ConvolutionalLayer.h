#ifndef CNN_GPU_CONVOLUTIONALLAYER_H
#define CNN_GPU_CONVOLUTIONALLAYER_H

#include "Layer.h"

class ConvolutionalLayer : public Layer {
private:
    size_t filterSize;
    size_t stride;
    size_t numFilters;

    std::vector<arma::cube> filters;

    double _getTruncNormalVal(double mean, double variance);

    void _resetAccumulatedGradients();

    std::vector<arma::cube> gradFilters;
    std::vector<arma::cube> accumulatedGradFilters;

public:
    ConvolutionalLayer(size_t numFilters, size_t filterSize, size_t stride);

    void init() override;

    void feedForward() override;

    void backPropagate() override;

    void UpdateFilterWeights(size_t batchSize, double learningRate);

    void setFilters(std::vector<arma::cube> filters);

    std::vector<arma::cube> getFilters();

    std::vector<arma::cube> getGradientWrtFilters();
};

#endif //CNN_GPU_CONVOLUTIONALLAYER_H
