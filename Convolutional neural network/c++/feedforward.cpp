#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <random>
#include "matrixlib.cpp"
#include "kernel.cpp"


int main(){


    int ImgHeight = 28;
    int ImgWidth = 28;
    Matrix Image(ImgHeight, ImgWidth);
    std::ifstream file;
    file.open("img/2_1.pgm", std::ios::in | std::ios::binary);
    if (!file.is_open()) {
		std::cout << "Error in open file" << std::endl;
		return 1;
	}
    
    unsigned char data;
    for(int k = 0; k < ImgHeight; k++)
    {
        for(int j = 0; j < ImgWidth; j++)
        {
            file.read((char*) &data, sizeof(data));
            Image.data[k][j] = (double)data;
        }
    }
    file.close();
    

    //Image.print();

    Matrix mtx;
    mtx.rows = 5;
    mtx.cols = 5;
    mtx.data = {{1,1,1,1,1},{0,0,0,0,0},{0,0,0,0,0},{-1,-1,-1,-1,-1},{-1,-1,-1,-1,-1}};


    Kernel kernel(Image, mtx);
    kernel.pool(6);
    kernel.result.print();

    displaying and normalizing results

    int biggest = 0;
    for(int k = 0; k < kernel.result.rows; k++)
    {
        for(int j = 0; j < kernel.result.cols; j++)
        {
            biggest = (kernel.result.data[k][j] > biggest) ? kernel.result.data[k][j] : biggest;
        }
    }
    Matrix normalized(kernel.result.rows, kernel.result.cols);
    for(int k = 0; k < kernel.result.rows; k++)
    {
        for(int j = 0; j < kernel.result.cols; j++)
        {
            normalized.data[k][j] = (int)(kernel.result.data[k][j] / biggest *255);
        }
    }
    //normalized.print();
    
    // std::ofstream output;
    // output.open("img_processed/2_1.pgm", std::ios::out | std::ios::binary);
    // if (!output.is_open()) {
	// 	std::cout << "Error in open file" << std::endl;
	// 	return 1;
	// }
    // output << "P5\n";
    // output << normalized.rows <<"\n";
    // output << normalized.cols <<"\n";
    // output << "255\n";
    // for(int k = 0; k < normalized.rows; k++)
    // {
    //     for(int j = 0; j < normalized.cols; j++)
    //     {
    //         unsigned char data = normalized.data[k][j];
    //         //std::cout << (int) data << std::endl;
    //         output.write((char*) &data, sizeof(data));
    //     }
    // }
    // output.close();






    return 0;
}