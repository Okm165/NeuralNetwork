#pragma once

#define imgf "train-images.idx3-ubyte"
#define labelf "train-labels.idx1-ubyte"
#define WIDTH 28
#define HEIGHT 28
#define NUM 200//60000

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
				img.data[k][j] = m;
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
	return result;
}