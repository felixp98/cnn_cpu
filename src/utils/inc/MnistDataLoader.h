//
// Created by felix on 24.10.18.
//

#ifndef CNN_GPU_MNISTDATALOADER_H
#define CNN_GPU_MNISTDATALOADER_H

#include <armadillo>
#include <list>
#include <iostream>

using namespace std;

class MnistDataLoader {
private:
    //std::list<arma::cube> trainData;
    //std::list<arma::cube> validationData;
    //std::list<arma::cube> testData;

public:
    std::list<arma::cube> readMnistTrainingData(){
        std::list<arma::cube> trainingData;
        ifstream file ("C:\\t10k-images.idx3-ubyte",ios::binary);
        if (file.is_open())
        {
            int magic_number=0;
            int number_of_images=0;
            int n_rows=0;
            int n_cols=0;
            file.read((char*)&magic_number,sizeof(magic_number));
            magic_number= ReverseInt(magic_number);
            file.read((char*)&number_of_images,sizeof(number_of_images));
            number_of_images= ReverseInt(number_of_images);
            file.read((char*)&n_rows,sizeof(n_rows));
            n_rows= ReverseInt(n_rows);
            file.read((char*)&n_cols,sizeof(n_cols));
            n_cols= ReverseInt(n_cols);
            for(int i=0;i<number_of_images;++i)
            {
                arma::cube tmpCube = arma::zeros(32, 32, 1);
                for(int r=0;r<n_rows;++r)
                {
                    for(int c=0;c<n_cols;++c)
                    {
                        unsigned char temp=0;
                        file.read((char*)&temp,sizeof(temp));

                        tmpCube.at
                        arr[i][(n_rows*r)+c]= (double)temp;
                    }
                }
            }
        }
    }

    std::list<arma::cube> readMnistTestData(){

    }

    void ReadMNIST(int NumberOfImages, int DataOfAnImage,vector<vector<double>> &arr)
    {
        arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
        ifstream file ("C:\\t10k-images.idx3-ubyte",ios::binary);
        if (file.is_open())
        {
            int magic_number=0;
            int number_of_images=0;
            int n_rows=0;
            int n_cols=0;
            file.read((char*)&magic_number,sizeof(magic_number));
            magic_number= ReverseInt(magic_number);
            file.read((char*)&number_of_images,sizeof(number_of_images));
            number_of_images= ReverseInt(number_of_images);
            file.read((char*)&n_rows,sizeof(n_rows));
            n_rows= ReverseInt(n_rows);
            file.read((char*)&n_cols,sizeof(n_cols));
            n_cols= ReverseInt(n_cols);
            for(int i=0;i<number_of_images;++i)
            {
                for(int r=0;r<n_rows;++r)
                {
                    for(int c=0;c<n_cols;++c)
                    {
                        unsigned char temp=0;
                        file.read((char*)&temp,sizeof(temp));
                        arr[i][(n_rows*r)+c]= (double)temp;
                    }
                }
            }
        }
    }

    int ReverseInt (int i) {
        unsigned char ch1, ch2, ch3, ch4;
        ch1 = i & 255;
        ch2 = (i >> 8) & 255;
        ch3 = (i >> 16) & 255;
        ch4 = (i >> 24) & 255;
    }
};


#endif //CNN_GPU_MNISTDATALOADER_H
