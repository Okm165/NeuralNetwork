#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <chrono>
#include <fstream>
#include <mutex>
#include <thread>
#include <pthread.h>

#define imgf "train-images.idx3-ubyte"
#define labelf "train-labels.idx1-ubyte"
#define WIDTH 28
#define HEIGHT 28
#define NUM 60000//60000 //500
#define GAPSIZE 1000

//#define NN { 80,1040,840,640,480,350,250,150,70,10 }

std::vector<int> structure = { 80,100,100,80,10 };
int learningLength = 45000000;
double learningRate = 0.001;

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
	static Matrix scale(Matrix& mtx, int& layer)
    {
        Matrix result(mtx.rows, mtx.cols);
        for (int k = 0; k < mtx.rows; k++)
            for (int j = 0; j < mtx.cols; j++)
                result.data[k][j] = mtx.data[k][j]/(sqrt(layer));
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
    return x*x;
}
int doneINT = 0;
void progressBar()
{
    doneINT++;
    std::cout << "\r" << doneINT << "          " << std::flush;
}

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

typedef std::vector<Matrix> Matrix_Array;

struct feedForwardReturnObject
{
	Matrix_Array ZArray;
	Matrix_Array AArray;
	std::vector<double> guess;
};

struct gradientsMatricesReturnObject
{
	Matrix_Array weightsMatricesFinal;
	Matrix_Array biasesMatricesFinal;
	double costFunction;
};

std::vector<double> costFunctionLog;

class NeuralNetwork
{
public:
	std::vector<int> layers;
	Matrix_Array weightMatrices;
	Matrix_Array biasMatrices;
	NeuralNetwork(const std::vector<int>& data)
	{
		layers = data;
		for (unsigned int p = 0; p < layers.size()-1; p++)
		{
			Matrix weightMatrix(layers[p + 1], layers[p]);
			weightMatrix.randomize();
			weightMatrix = Matrix::scale(weightMatrix, layers[p]);
			weightMatrices.emplace_back(weightMatrix);

			Matrix biasMatrix(layers[p + 1], 1);
			biasMatrix.randomize();
			biasMatrices.emplace_back(biasMatrix);
		}
	}
	feedForwardReturnObject feedForward(std::vector<double>& input)
	{
		Matrix_Array ZArray, AArray;
		Matrix temporaryResult = Matrix::toMatrix(input);
		feedForwardReturnObject result;
		for (unsigned int p = 0; p < this->layers.size() - 1; p++)
		{
			temporaryResult = this->weightMatrices[p] * temporaryResult + this->biasMatrices[p];
			ZArray.emplace_back(temporaryResult);
			temporaryResult = Matrix::map(temporaryResult, activ);
			AArray.emplace_back(temporaryResult);
		}
		result.ZArray = Matrix::reverseArray(ZArray);
		result.AArray = Matrix::reverseArray(AArray);
		result.guess = Matrix::toArray(temporaryResult);
		return result;
	}
	gradientsMatricesReturnObject gradients(std::vector<double>& input, std::vector<double>& target, double& learningRate)
	{
		Matrix_Array RWeightMatrices, RBiasMatrices, weightsGradMatrices, biasGradMatrices;
		gradientsMatricesReturnObject result;
		feedForwardReturnObject feedForward = this->feedForward(input);
		RWeightMatrices = Matrix::reverseArray(this->weightMatrices);
		RBiasMatrices = Matrix::reverseArray(this->biasMatrices);
		std::vector<int> Rlayers = Matrix::reverseArray(this->layers);
		feedForward.AArray.emplace_back(Matrix::toMatrix(input));
		Matrix a = (feedForward.AArray[0] + Matrix::toMatrix(target) * -1);
		Matrix miuMatrix = a * 2;
		for (unsigned int p = 0; p < layers.size() - 1; p++)
		{
			Matrix weightGradMatrix(Rlayers[p], Rlayers[p + 1]);
			for (int k = 0; k < Rlayers[p]; k++)
				for (int j = 0; j < Rlayers[p + 1]; j++)
					weightGradMatrix.data[k][j] = feedForward.AArray[p + 1].data[j][0] * activDeriv(feedForward.ZArray[p].data[k][0]) * miuMatrix.data[k][0];

			Matrix biasGradMatrix(Rlayers[p], 1);
			for (int k = 0; k < Rlayers[p]; k++)
				biasGradMatrix.data[k][0] = activDeriv(feedForward.ZArray[p].data[k][0]) * miuMatrix.data[k][0];

			Matrix newMiuMatrix(Rlayers[p + 1], 1);
			for (int k = 0; k < Rlayers[p + 1]; k++)
			{
				double sum = 0;
				for (int o = 0; o < Rlayers[p]; o++)
				{
					sum += miuMatrix.data[o][0] * activDeriv(feedForward.ZArray[p].data[o][0]) * RWeightMatrices[p].data[o][k];
				}
				newMiuMatrix.data[k][0] = sum;
			}
			miuMatrix = newMiuMatrix;
			result.weightsMatricesFinal.emplace_back(RWeightMatrices[p] + (weightGradMatrix * -1) * learningRate);
			result.biasesMatricesFinal.emplace_back(RBiasMatrices[p] + (biasGradMatrix * -1) * learningRate);
		}
		result.costFunction = Matrix::collapse(Matrix::map(a, quad));
		result.weightsMatricesFinal = Matrix::reverseArray(result.weightsMatricesFinal);
		result.biasesMatricesFinal = Matrix::reverseArray(result.biasesMatricesFinal);
		return result;
	}
	void learn(gradientsMatricesReturnObject& matricesFinal)
	{
		this->weightMatrices = matricesFinal.weightsMatricesFinal;
		this->biasMatrices = matricesFinal.biasesMatricesFinal;
	}
};

struct loadDataReturnObject
{
	unsigned char* targets_pointer;
	int targets_pointer_size;
	Matrix* images_pointer;
};

struct dataPrepReturnObject
{
	MATRIX target_array;
	MATRIX input_array;
};

loadDataReturnObject loadData()
{
	loadDataReturnObject result;
	result.targets_pointer = new unsigned char[NUM];
	result.images_pointer = new Matrix[NUM];
	result.targets_pointer_size = NUM;
	std::fstream datafile, labelfile;
	datafile.open(imgf, std::ios::in | std::ios::binary);
	labelfile.open(labelf, std::ios::in | std::ios::binary);
	if (!datafile || !labelfile) { std::cout << "opening file failed" << std::endl; };
	datafile.seekg(16, std::ios::beg);
	labelfile.seekg(8, std::ios::beg);
	for (unsigned int i = 0; i < NUM; i++)
	{
		unsigned char l;
		labelfile.read((char*)&l, 1);
		result.targets_pointer[i] = l;

		Matrix img(HEIGHT, WIDTH);
		for (unsigned int k = 0; k < (HEIGHT); k++)
			for (unsigned int j = 0; j < (WIDTH); j++)
			{
				unsigned char m;
				datafile.read((char*)&m, 1);
				img.data[k][j] = m/255.0;
			}
		result.images_pointer[i] = img;
	}
	return result;
}

Matrix_Array filterPrep()
{
	Matrix botEdgeFilter;
	botEdgeFilter.rows = 5;
	botEdgeFilter.cols = 5;
	botEdgeFilter.data = { {1,1,1,1,1},{0,0,0,0,0},{0,0,0,0,0},{-1,-1,-1,-1,-1},{-1,-1,-1,-1,-1} };

	Matrix topEdgeFilter;
	topEdgeFilter.rows = 5;
	topEdgeFilter.cols = 5;
	topEdgeFilter.data = { {-1,-1,-1,-1,-1},{-1,-1,-1,-1,-1},{0,0,0,0,0},{0,0,0,0,0},{1,1,1,1,1} };

	Matrix leftEdgeFilter;
	leftEdgeFilter.rows = 5;
	leftEdgeFilter.cols = 5;
	leftEdgeFilter.data = { {1,0,0,-1,-1},{1,0,0,-1,-1},{1,0,0,-1,-1},{1,0,0,-1,-1},{1,0,0,-1,-1} };

	Matrix rightEdgeFilter;
	rightEdgeFilter.rows = 5;
	rightEdgeFilter.cols = 5;
	rightEdgeFilter.data = { {-1,-1,0,0,1},{-1,-1,0,0,1},{-1,-1,0,0,1},{-1,-1,0,0,1},{-1,-1,0,0,1} };

	Matrix centerEdgeFilter;
	centerEdgeFilter.rows = 5;
	centerEdgeFilter.cols = 5;
	centerEdgeFilter.data = { {-1,0,1,0,-1},{0,1,0,1,0},{1,0,-1,0,1},{0,1,0,1,0},{-1,0,1,0,-1} };

	Matrix_Array array;
	array.emplace_back(botEdgeFilter);
	array.emplace_back(topEdgeFilter);
	array.emplace_back(leftEdgeFilter);
	array.emplace_back(rightEdgeFilter);
	array.emplace_back(centerEdgeFilter);

	return array;
}

dataPrepReturnObject dataPrep()
{
	Matrix_Array filters = filterPrep();
	dataPrepReturnObject result;
	loadDataReturnObject data = loadData();
	for (int a = 0; a < data.targets_pointer_size; a++)
	{
		result.input_array.emplace_back();
		result.target_array.emplace_back();
		unsigned char b = data.targets_pointer[a];
		for (int g = 0; g < 10; g++)
		{
			if(g == (int)b)
				result.target_array[a].emplace_back(1);
			else
				result.target_array[a].emplace_back(0);
		}
		Matrix c = data.images_pointer[a];

		for (unsigned int o = 0; o < filters.size(); o++)
		{
			Kernel kernel(c, filters[o]);
			kernel.pool(6);
			for (int k = 0; k < kernel.result.rows; k++)
				for (int j = 0; j < kernel.result.cols; j++)
				{
					result.input_array[a].emplace_back(kernel.result.data[k][j]);
				}
		}
	}
	delete [] data.images_pointer;
	delete [] data.targets_pointer;
	return result;
}


std::mutex mtx;

void threadFunction(int& iterations, NeuralNetwork& nn, MATRIX& inputs, MATRIX& targets, double LearningRate, std::uniform_int_distribution<>& distrib2)
{
	for (int o = 0; o < iterations; o++)
	{
		int randomInputId = distrib2(gen);
		std::vector<double> currInput{ inputs[randomInputId] };
		std::vector<double> currTarget{ targets[randomInputId] };
		mtx.lock();
		NeuralNetwork localnn = nn;
		mtx.unlock();
		gradientsMatricesReturnObject gradientsMatrices = localnn.gradients(currInput, currTarget, LearningRate);
		mtx.lock();
		nn.learn(gradientsMatrices);
		costFunctionLog.emplace_back(gradientsMatrices.costFunction);
		mtx.unlock();
		progressBar();
	}
}

int main()
{
	std::cout << "\rpreparing training data..." << std::flush;
	dataPrepReturnObject result = dataPrep();
	std::cout << "\rtraining data prepared      " << std::endl;
	std::cout << "\rpreparing neural network..." << std::flush;
	NeuralNetwork nn(structure);
	std::cout << "\rneural network prepared                  " << std::endl;

	std::uniform_int_distribution<> distrib2(0, result.input_array.size() - 1);
	auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "training in progress..." << std::endl;
	std::cout << "\r" << "        %" << std::flush;

	const auto core_number = std::thread::hardware_concurrency();
	int division = learningLength / core_number;
	std::vector<int> core_iterations (core_number, division);
	int remainder = learningLength % core_number;
	for (int i = 0; i < remainder; i++)
	{
		core_iterations[i]++;
	}
	std::vector<std::thread> Threads;

	for (unsigned int p = 0; p < core_number; p++)
	{
		Threads.emplace_back(std::thread(threadFunction, std::ref(core_iterations[p]), std::ref(nn), std::ref(result.input_array), std::ref(result.target_array), learningRate, std::ref(distrib2)));
	}
	for (auto& th : Threads)
	{
		th.join();
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> ms_double = t2 - t1;
	std::cout << "timing: " << std::endl;
	std::cout << ms_double.count() << " ms" << std::endl;
	std::cout << std::endl;

	std::cout << "neural network trained" << std::endl;
	std::cout << std::endl;

	std::cout << "saving neural network..." << std::flush;
	std::ofstream out;
	out.open("neuralnetwork.txt");
	// header
	out << "neural network body\n";
	out << "structure:\n";
	for (unsigned int a = 0; a < structure.size(); a++)
		out << structure[a] << ",";
	out << "\n";
	out << "weights:";
	for (unsigned int p = 0; p < nn.weightMatrices.size(); p++)
		for (int k = 0; k < nn.weightMatrices[p].rows; k++)
			for (int j = 0; j < nn.weightMatrices[p].cols; j++)
				out << nn.weightMatrices[p].data[k][j] << ",";
	out << "\n";
	out << "biases:";
	for (unsigned int p = 0; p < nn.biasMatrices.size(); p++)
		for (int k = 0; k < nn.biasMatrices[p].rows; k++)
			out << nn.biasMatrices[p].data[k][0] << ",";
	out.close();
	std::cout << "\r" << "neural network saved in neuralnetwork.txt" << std::endl;


	std::cout << "\r" << "reducing cost log..." << std::flush;
	// averaging cost output
	std::vector<double> costFunctionLogReduced;
	int gap = costFunctionLog.size() / GAPSIZE;
	int gapRemainder = costFunctionLog.size() % GAPSIZE;
	for (int p = 0; p < gap; p++)
	{
		double sum = 0;
		for (int o = 0; o < GAPSIZE; o++)
		{
			sum += costFunctionLog[p*GAPSIZE + o];
		}
		costFunctionLogReduced.emplace_back(sum/GAPSIZE);
	}
	if (gapRemainder > 0)
	{
		double Rsum = 0;
		for (int p = 0; p < gapRemainder; p++)
		{

			Rsum += costFunctionLog[gap * GAPSIZE + p];

		}
		costFunctionLogReduced.emplace_back(Rsum / gapRemainder);
	}
	std::cout << "\r" << "cost log reduced          " << std::endl;




	//puting into file
	std::ofstream file;
	file.open("output.txt");
	for (unsigned int a = 0; a < costFunctionLogReduced.size(); a++)
		file << costFunctionLogReduced[a] << "\n";
	file.close();

	std::cout << "cost function data saved to output.txt" << std::endl;

	std::cin.get();
}
