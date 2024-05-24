void NeuralNetwork::train(vector<vector<double>>& inputs, vector<vector<double>>& outputs, int epochs, double learningRate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int i = 0; i < inputs.size(); ++i) {
            forward(inputs[i]);
            backward(inputs[i], outputs[i], learningRate);
        }
    }
}

vector<double> NeuralNetwork::predict(vector<double>& input) {
    forward(input);
    return layerOutputs.back();
}
