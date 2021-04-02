#pragma once
#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <chrono>
#include <fstream>

typedef std::vector<std::vector<double>> MATRIX;


std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> distrib(-1.0, 1.0);


class Matrix {
public:

    int rows, cols;
    MATRIX data;

    Matrix() {}
    Matrix(const int &rows, const int &cols)
    {
        this->rows = rows;
        this->cols = cols;
        this->data = MATRIX(rows, std::vector<double>(cols, 0));
    }
    void randomize()
    {
        for (int k = 0; k < this->rows; k++)
        {
            for (int j = 0; j < this->cols; j++)
            {
                this->data[k][j] = distrib(gen);
            }
        }
    }
    void print()
    {
        for (int k = 0; k < this->rows; k++)
        {
            for (int j = 0; j < this->cols; j++)
            {
                std::cout << this->data[k][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    Matrix operator + (const Matrix &mtx)
    {
        Matrix result(this->rows, this->cols);
        for (int k = 0; k < this->rows; k++)
            for (int j = 0; j < this->cols; j++)
            {
                result.data[k][j] = this->data[k][j] + mtx.data[k][j];
            }
        return result;
    }
    Matrix operator * (const Matrix &mtx)
    {
        Matrix result(this->rows, mtx.cols);
        for (int k = 0; k < result.rows; k++)
            for (int o = 0; o < result.cols; o++)
            {
                double sum = 0;
                for (int j = 0; j < this->cols; j++)
                {
                    sum += this->data[k][j] * mtx.data[j][o];
                }
                result.data[k][o] = sum;
            }
        return result;
    }
    Matrix operator * (const double& a)
    {
        Matrix result(this->rows, this->cols);
        for (int k = 0; k < result.rows; k++)
            for (int j = 0; j < result.cols; j++)
                result.data[k][j] = this->data[k][j] * a;
        return result;
    }
    static Matrix map(Matrix& mtx, double(*func)(double& x))
    {
        Matrix result(mtx.rows, mtx.cols);
        for (int k = 0; k < mtx.rows; k++)
            for (int j = 0; j < mtx.cols; j++)
                result.data[k][j] = func(mtx.data[k][j]);
        return result;
    }
    static Matrix toMatrix(std::vector<double> &a)
    {
        Matrix result(a.size(), 1);
        for (int k = 0; k < result.rows; k++)
            result.data[k][0] = a[k];

        return result;
    }
    static std::vector<double> toArray(Matrix &a)
    {
        std::vector<double> result;
        for (int k = 0; k < a.rows; k++)
            for (int j = 0; j < a.cols; j++)
                result.emplace_back(a.data[k][j]);
        return result;
    }
    static std::vector<Matrix> reverseArray(std::vector<Matrix>&a)
    {
        std::vector<Matrix> result;
        for (unsigned int o = 0; o < a.size(); o++)
        {
            int counter = a.size() - o - 1;
            result.emplace_back(a[counter]);
        }
        return result;
    }
    static std::vector<int> reverseArray(std::vector<int> &a)
    {
        std::vector<int> result;
        for (unsigned int o = 0; o < a.size(); o++)
        {
            int counter = a.size() - o - 1;
            result.emplace_back(a[counter]);
        }
        return result;
    }
    static double collapse(const Matrix& mtx)
    {
        double result = 0;
        for (int k = 0; k < mtx.rows; k++)
            for (int j = 0; j < mtx.cols; j++)
                result += mtx.data[k][j];
        return result;
    }
};
//fast sigmoid algorythm
double activ(double& x) {
    return 1 / (1 + exp(-x));
}
double activDeriv(double& x) {
    return activ(x) * (1 - activ(x));
}
double quad(double& x) {
    return x * x;
}