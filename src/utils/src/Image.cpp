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

string &Image::getLabel() {
    return label;
}

void Image::setLabel(string &label) {
    Image::label = label;
}

arma::cube & Image::getImageData() {
    return imageData;
}

void Image::setImageData(arma::cube &imageData) {
    Image::imageData = imageData;
}

