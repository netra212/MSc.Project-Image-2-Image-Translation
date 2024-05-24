int main() {
    vector<int> architecture = {2, 2, 1};
    NeuralNetwork nn(architecture);

    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> outputs = {{0}, {1}, {1}, {0}};

    nn.train(inputs, outputs, 10000, 0.1);

    for (auto& input : inputs) {
        vector<double> output = nn.predict(input);
        cout << input[0] << " XOR " << input[1] << " = " << output[0] << endl;
    }

    return 0;
}
