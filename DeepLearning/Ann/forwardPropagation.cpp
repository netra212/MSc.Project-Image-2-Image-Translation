#include 'annStructure'

void NeuralNetwork::forward(vector<double>& input) {
    layerOutputs[0] = input;
    for (int i = 0; i < architecture.size() - 1; ++i) {
        for (int j = 0; j < architecture[i + 1]; ++j) {
            double activation = biases[i][j];
            for (int k = 0; k < architecture[i]; ++k) {
                activation += weights[i][j][k] * layerOutputs[i][k];
            }
            layerOutputs[i + 1][j] = sigmoid(activation);
        }
    }
}
