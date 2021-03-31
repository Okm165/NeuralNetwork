#pragma once
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
};
std::vector<double> data;

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
	gradientsMatricesReturnObject gradients(std::vector<double>& input, std::vector<double>& target)
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
		//---cost logging
		data.emplace_back(Matrix::collapse(Matrix::map(a, quad)));
		//---
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
			result.weightsMatricesFinal.emplace_back(RWeightMatrices[p] + weightGradMatrix * -1);
			result.biasesMatricesFinal.emplace_back(RBiasMatrices[p] + biasGradMatrix * -1);
		}

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