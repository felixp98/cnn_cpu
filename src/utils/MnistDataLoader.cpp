//
// Created by felix on 24.10.18.
//

#include "utils/MnistDataLoader.h"
#include "MnistDataLoader.h"


MnistDataLoader::MnistDataLoader(string pathToData) {

    this->pathToData = pathToData;
    pathToTrainData = pathToData + "/train-images-idx3-ubyte";
    pathToTrainLabels = pathToData + "/train-labels-idx1-ubyte";
    pathToTestData = pathToData + "/t10k-images-idx3-ubyte";
    pathToTestLabels = pathToData + "/t10k-labels-idx1-ubyte";

    trainData = readMnistData(pathToTrainData, pathToTrainLabels);
    testData = readMnistData(pathToTestData, pathToTestLabels);

    splitTrainDataToValidationData(0.9, trainData, validationData);
}

std::vector<Image*> MnistDataLoader::readMnistData(string pathToImageFile, string pathToLabelFile){
    std::vector<Image*> data;
    ifstream imageFile (pathToImageFile, ios::binary);
    if (imageFile.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        imageFile.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        if(magic_number != 2051){std::cout << "Wrong magic number at Image File!" <<std::endl;}
        imageFile.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        imageFile.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        imageFile.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            arma::cube tmpCube = arma::zeros(28, 28, 1);
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    imageFile.read((char*)&temp,sizeof(temp));

                    tmpCube(r, c, 0) = temp/255.0;
                }
            }
            data.push_back(new Image(tmpCube));
        }
        imageFile.close();
    }

    ifstream labelFile (pathToLabelFile, ios::binary);
    if (labelFile.is_open())
    {
        int magic_number=0;
        int number_of_labels=0;
        labelFile.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        if(magic_number != 2049){std::cout << "Wrong magic number at Label File!" <<std::endl;}
        labelFile.read((char*)&number_of_labels,sizeof(number_of_labels));
        number_of_labels= ReverseInt(number_of_labels);

        for(int i=0;i<number_of_labels;++i)
        {
            unsigned char temp=0;
            labelFile.read((char*)&temp,sizeof(temp));
            string label = std::to_string(temp);

            arma::vec expectedScore = arma::zeros(10);
            expectedScore(temp) = 1.0;

            data.at(i)->setExpectedScore(expectedScore);
            data.at(i)->setLabel(label);
            data.at(i)->setNumClasses(10);
        }
        labelFile.close();
    }

    return data;
}

int MnistDataLoader::ReverseInt (int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void MnistDataLoader::splitTrainDataToValidationData(double splitRatio, vector<Image *> &trainData,
                                                     vector<Image *> &validationData) {
    if(trainData.size() == 0 || splitRatio <= 0.0  || splitRatio >= 1.0){
        //Error, train Data is not loaded yet
        return;
    }

    arma::vec randomIdxs(trainData.size()*(1-splitRatio), arma::fill::randu);
    randomIdxs *= (trainData.size() - 1);

    for(size_t i=0; i<randomIdxs.size(); ++i){
        validationData.push_back(trainData[i]);
    }
    for(size_t i=0; i<randomIdxs.size(); ++i){
        //delete validation data from train data vector from highest index to lowest
        trainData.erase(trainData.begin() + (int)randomIdxs.max());
        randomIdxs[randomIdxs.index_max()] = -1.0;
    }
}

const vector<Image *> &MnistDataLoader::getTrainData() const {
    return trainData;
}

const vector<Image *> &MnistDataLoader::getValidationData() const {
    return validationData;
}

const vector<Image *> &MnistDataLoader::getTestData() const {
    return testData;
}
