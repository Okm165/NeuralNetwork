#include "matrixlib.h"
#include "kernel.h"
#include "neuralNetwork.h"

int main()
{
	NeuralNetwork nn({2,4,2});
	int learningLength = 10000;
	MATRIX input_array = { {1,1},{0,0},{0.5,0.5},{0.7,0.1} };
	MATRIX target_array = { {0,0},{1,1},{1,0},{0,1} };
	std::uniform_int_distribution<> distrib2(0, input_array.size()-1);
	auto t1 = std::chrono::high_resolution_clock::now();
	for (int a = 0; a < learningLength; a++)
	{
		int random = distrib2(gen);
		std::vector<double> currInput = input_array[random];
		std::vector<double> currTarget = target_array[random];
		gradientsMatricesReturnObject gradients = nn.gradients(currInput, currTarget);
		nn.learn(gradients);
		
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> ms_double = t2 - t1;
	std::cout << "timing: " << std::endl;
	std::cout << ms_double.count() << " ms" << std::endl;
	std::cout << std::endl;
	//display results

	for (unsigned int b = 0; b < input_array.size(); b++)
	{
		std::vector<double> currInput = input_array[b];
		feedForwardReturnObject fed = nn.feedForward(currInput);
		for (unsigned int a = 0; a < fed.guess.size(); a++)
		{
			std::cout << fed.guess[a] << " ";
		}
		std::cout << std::endl;
	}
	//puting into file
	std::ofstream file;
	file.open("output.txt");
	for (unsigned int a = 0; a < data.size(); a++)
		file << data[a] << "\n";
	file.close();


	std::cin.get(); 
}