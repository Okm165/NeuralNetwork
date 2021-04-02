#include "matrixlib.h"
#include "kernel.h"
#include "neuralNetwork.h"
#include "dataprep.h"

int main()
{
	std::cout << "\rpreparing training data..." << std::flush;
	dataPrepReturnObject result = dataPrep();
	std::cout << "\rtraining data prepared      " << std::endl;
	std::cout << "\rpreparing neural network..." << std::flush;
	NeuralNetwork nn({ 80,150,100,50,10 });
	std::cout << "\rneural network prepared                  " << std::endl;
	int learningLength = 40000;
	double learningRate = 0.05;
	std::uniform_int_distribution<> distrib2(0, result.input_array.size() - 1);
	auto t1 = std::chrono::high_resolution_clock::now();
	std::cout << "training in progress..." << std::endl;
	for (int a = 0; a < learningLength; a++)
	{
		int random = distrib2(gen);
		std::vector<double> currInput = result.input_array[random];
		std::vector<double> currTarget = result.target_array[random];
		gradientsMatricesReturnObject gradients = nn.gradients(currInput, currTarget, learningRate);
		nn.learn(gradients);
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> ms_double = t2 - t1;
	std::cout << "timing: " << std::endl;
	std::cout << ms_double.count() << " ms" << std::endl;
	std::cout << std::endl;

	/*for (unsigned int b = 0; b < result.input_array.size(); b++)
	{
		std::vector<double> currInput = result.input_array[b];
		feedForwardReturnObject fed = nn.feedForward(currInput);
		for (unsigned int a = 0; a < fed.guess.size(); a++)
		{
			std::cout << fed.guess[a] << " ";
		}
		std::cout << std::endl;
	}*/

	//puting into file
	std::ofstream file;
	file.open("output.txt");
	for (unsigned int a = 0; a < data.size(); a++)
		file << data[a] << "\n";
	file.close();
	std::cout << "neural network trained" << std::endl;
	std::cout << std::endl;
	std::cout << "cost function data saved to output.txt" << std::endl;



















	/*for (unsigned int g = 0; g < result.target_array.size(); g++)
	{
		for (unsigned int h = 0; h < result.target_array[g].size(); h++)
		{
			std::cout << result.target_array[g][h] << " ";
		}
		std::cout << std::endl;
	}*/






	//NeuralNetwork nn({2,4,2});
	//int learningLength = 10000;
	//MATRIX input_array = { {1,1},{0,0},{0.5,0.5},{0.7,0.1} };
	//MATRIX target_array = { {0,0},{1,1},{1,0},{0,1} };
	//std::uniform_int_distribution<> distrib2(0, input_array.size()-1);
	//auto t1 = std::chrono::high_resolution_clock::now();
	//for (int a = 0; a < learningLength; a++)
	//{
	//	int random = distrib2(gen);
	//	std::vector<double> currInput = input_array[random];
	//	std::vector<double> currTarget = target_array[random];
	//	gradientsMatricesReturnObject gradients = nn.gradients(currInput, currTarget);
	//	nn.learn(gradients);
	//	
	//}
	//auto t2 = std::chrono::high_resolution_clock::now();
	//std::chrono::duration<double, std::milli> ms_double = t2 - t1;
	//std::cout << "timing: " << std::endl;
	//std::cout << ms_double.count() << " ms" << std::endl;
	//std::cout << std::endl;
	////display results

	//for (unsigned int b = 0; b < input_array.size(); b++)
	//{
	//	std::vector<double> currInput = input_array[b];
	//	feedForwardReturnObject fed = nn.feedForward(currInput);
	//	for (unsigned int a = 0; a < fed.guess.size(); a++)
	//	{
	//		std::cout << fed.guess[a] << " ";
	//	}
	//	std::cout << std::endl;
	//}
	////puting into file
	//std::ofstream file;
	//file.open("output.txt");
	//for (unsigned int a = 0; a < data.size(); a++)
	//	file << data[a] << "\n";
	//file.close();


	std::cin.get(); 
}