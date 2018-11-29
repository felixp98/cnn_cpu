//
// Created by felix on 30.10.18.
//

#ifndef CNN_GPU_IMAGE_H
#define CNN_GPU_IMAGE_H

#include <string>
#include <armadillo>

using namespace std;

class Image{
private:
    string label = "";
    int classIndex;
    int numClasses;
    arma::vec expectedScore;
    arma::cube imageData;

public:
    Image();
    Image(string label);
    Image(arma::cube imageData);
    Image(string label, arma::cube imageData);

    arma::vec& getExpectedScore();

    string &getLabel();

    void setLabel(string &label);

    arma::cube & getImageData();

    void setImageData(arma::cube &imageData);

    int getClassIndex();

    void setClassIndex(int classIndex);

    int getNumClasses();

    void setNumClasses(int numClasses);
};

#endif //CNN_GPU_IMAGE_H
