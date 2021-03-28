typedef std::vector<std::vector<double>> MATRIX;
typedef std::vector<MATRIX> MATRIX_ARRAY;

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> distrib(-1.0, 1.0);

class Matrix{
    public:

    int rows, cols;
    MATRIX data;

    Matrix()
    {

    }
    Matrix(int rows, int cols)
    {
        this->rows = rows;
        this->cols = cols;
        this->data = MATRIX(rows, std::vector<double> (cols, 0));
    }
    void randomize()
    {
        for(int k = 0; k < this->rows; k++)
        {
            for(int j = 0; j < this->cols; j++)
            {
                this->data[k][j] = distrib(gen);
            }
        }
    }
    void print()
    {
        for(int k = 0; k < this->rows; k++)
        {
            for(int j = 0; j < this->cols; j++)
            {
                std::cout << this->data[k][j] << " ";
            }
            std::cout << std::endl;
        }
    }
};
//fast sigmoid algorythm
double activ(double &x){
    return (tanh(x)+1)/2;
} 
double activDeriv(double &x){
    return (1-tanh(x)*tanh(x))/2;
}