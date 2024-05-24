#include 'annStructure.h'

void NeuralNetwork::backward(vector<double>& input, vector<double>& target, double learningRate) {
    vector<double> errors = layerOutputs.back();
    for (int i = 0; i < target.size(); ++i) {
        errors[i] = target[i] - errors[i];
    }

    for (int i = architecture.size() - 2; i >= 0; --i) {
        vector<double> layerError(architecture[i + 1]);
        vector<double> nextLayerDelta = (i == architecture.size() - 2) ? errors : layerDeltas[i + 1];
        for (int j = 0; j < architecture[i + 1]; ++j) {
            double delta = nextLayerDelta[j] * sigmoidDerivative(layerOutputs[i + 1][j]);
            layerDeltas[i][j] = delta;
            for (int k = 0; k < architecture[i]; ++k) {
                weights[i][j][k] += learningRate * delta * layerOutputs[i][k];
            }
            biases[i][j] += learningRate * delta;
        }
    }
}
