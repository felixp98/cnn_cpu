//
// Created by felix on 30.10.18.
//

#include <utils/inc/Image.h>

Image::Image() = default;

Image::Image(string label) {
    this->label = label;
}

Image::Image(arma::cube imageData) {
    this->imageData = imageData;
}

Image::Image(string label, arma::cube imageData) {
    this->label = label;
    this->imageData = imageData;
}

const string &Image::getLabel() const {
    return label;
}

void Image::setLabel(const string &label) {
    Image::label = label;
}

const arma::cube &Image::getImageData() const {
    return imageData;
}

void Image::setImageData(const arma::cube &imageData) {
    Image::imageData = imageData;
}

