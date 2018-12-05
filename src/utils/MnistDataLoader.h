//
// Created by felix on 24.10.18.
//

#ifndef CNN_GPU_MNISTDATALOADER_H
#define CNN_GPU_MNISTDATALOADER_H

#include <armadillo>
#include <iostream>
#include "Image.h"
#include <string>

using namespace std;

class MnistDataLoader {
private:

    string pathToData;
    string pathToTrainData;
    string pathToTrainLabels;
    string pathToTestData;
    string pathToTestLabels;

    vector<Image*> trainData;
    vector<Image*> validationData;
    vector<Image*> testData;

    std::vector<Image*> readMnistData(string pathToImageFile, string pathToLabelFile);
    int ReverseInt(int i);
    void splitTrainDataToValidationData(double splitRatio, vector<Image*> &trainData, vector<Image*> &validationData);

public:
    explicit MnistDataLoader(string relativePathToData, double splitRatio);

    const vector<Image *> &getTrainData() const;

    const vector<Image *> &getValidationData() const;

    const vector<Image *> &getTestData() const;
};


#endif //CNN_GPU_MNISTDATALOADER_H
