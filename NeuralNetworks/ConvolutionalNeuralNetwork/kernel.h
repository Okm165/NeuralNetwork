#pragma once
class Kernel
{
public:

    Matrix result;

    Kernel(Matrix &Image, Matrix &Filter)
    {
        result = Matrix(Image.rows - Filter.rows + 1, Image.cols - Filter.cols + 1);
        for (int k = 0; k < result.rows; k++)
        {
            for (int j = 0; j < result.cols; j++)
            {
                double sum = 0;
                for (int lokalK = 0; lokalK < Filter.rows; lokalK++)
                {
                    for (int lokalJ = 0; lokalJ < Filter.cols; lokalJ++)
                    {
                        sum += Image.data[k + lokalK][j + lokalJ] * Filter.data[lokalK][lokalJ];
                    }
                }
                result.data[k][j] = std::max(0.0, sum);
            }
        }
    }
    void pool(int step) {
        Matrix pooled(this->result.rows / step, this->result.cols / step);
        for (int k = 0; k < this->result.rows; k += step)
        {
            for (int j = 0; j < this->result.cols; j += step)
            {
                std::vector<double> sum;
                for (int lokalK = 0; lokalK < step - 1; lokalK++)
                {
                    for (int lokalJ = 0; lokalJ < step - 1; lokalJ++)
                    {
                        sum.emplace_back(this->result.data[k + lokalK][j + lokalJ]);
                    }
                }
                double biggest = 0;
                for (unsigned int k = 0; k < sum.size(); k++)
                {
                    biggest = (sum[k] > biggest) ? sum[k] : biggest;
                }
                pooled.data[k / step][j / step] = biggest;

            }
        }
        this->result = pooled;
    }
};





