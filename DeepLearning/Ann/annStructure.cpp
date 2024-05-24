#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

class NeuralNetwork {
public:
    NeuralNetwork(vector<int> architecture);
    void train(vector<vector<double>>& inputs, vector<vector<double>>& outputs, int epochs, double learningRate);
    vector<double> predict(vector<double>& input);

private:
    vector<int> architecture;
    vector<vector<vector<double>>> weights;
    vector<vector<double>> biases;
    
    vector<double> activate(vector<double>& x);
    double sigmoid(double x);
    double sigmoidDerivative(double x);
    void forward(vector<double>& input);
    void backward(vector<double>& input, vector<double>& target, double learningRate);
    
    vector<vector<vector<double>>> layerOutputs;
    vector<vector<vector<double>>> layerDeltas;
};

NeuralNetwork::NeuralNetwork(vector<int> architecture) : architecture(architecture) {
    srand(time(0));
    weights.resize(architecture.size() - 1);
    biases.resize(architecture.size() - 1);
    layerOutputs.resize(architecture.size());
    layerDeltas.resize(architecture.size() - 1);

    for (int i = 0; i < architecture.size() - 1; ++i) {
        weights[i].resize(architecture[i + 1], vector<double>(architecture[i]));
        biases[i].resize(architecture[i + 1]);
        layerOutputs[i].resize(architecture[i]);
        
        for (int j = 0; j < architecture[i + 1]; ++j) {
            biases[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            for (int k = 0; k < architecture[i]; ++k) {
                weights[i][j][k] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }
    }
    layerOutputs[architecture.size() - 1].resize(architecture.back());
}

double NeuralNetwork::sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x) {
    return x * (1 - x);
}

vector<double> NeuralNetwork::activate(vector<double>& x) {
    vector<double> activated(x.size());
    for (int i = 0; i < x.size(); ++i) {
        activated[i] = sigmoid(x[i]);
    }
    return activated;
}
