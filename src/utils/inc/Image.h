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
    arma::cube imageData;

public:
    Image();
    Image(string label);
    Image(arma::cube imageData);
    Image(string label, arma::cube imageData);

    const string &getLabel() const;

    void setLabel(const string &label);

    const arma::cube &getImageData() const;

    void setImageData(const arma::cube &imageData);
};

#endif //CNN_GPU_IMAGE_H
