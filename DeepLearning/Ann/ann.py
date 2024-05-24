import numpy as np

class AdvancedANN:
    def __init__(self, layers, learning_rate=0.01):
        # Initialize network with given layer sizes and learning rate
        self.layers = layers  # List containing the number of neurons in each layer
        self.learning_rate = learning_rate  # Learning rate for gradient descent
        self.weights = []  # List to store weights for each layer
        self.biases = []  # List to store biases for each layer
        self.initialize_parameters()  # Initialize weights and biases

    def initialize_parameters(self):
        # Initialize weights and biases using He initialization
        for i in range(len(self.layers) - 1):
            # Initialize weights with random values scaled by He initialization
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i])
            # Initialize biases with zeros
            bias = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight)  # Append weights to the weights list
            self.biases.append(bias)  # Append biases to the biases list

    def relu(self, x):
        # ReLU activation function: max(0, x)
        return np.maximum(0, x)

    def relu_derivative(self, x):
        # Derivative of ReLU activation function
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        # Sigmoid activation function: 1 / (1 + exp(-x))
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivative of sigmoid activation function
        return x * (1 - x)

    def forward_propagation(self, X):
        # Forward propagation through the network
        activations = [X]  # List to store activations for each layer
        pre_activations = []  # List to store pre-activations (z values) for each layer

        for i in range(len(self.weights)):
            # Compute pre-activation z for the current layer
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)  # Store pre-activation z
            if i == len(self.weights) - 1:
                # Output layer uses sigmoid activation
                a = self.sigmoid(z)
            else:
                # Hidden layers use ReLU activation
                a = self.relu(z)
            activations.append(a)  # Store activation a

        return activations, pre_activations

    def backward_propagation(self, X, y, activations, pre_activations):
        # Backward propagation through the network
        m = y.shape[0]  # Number of training examples
        deltas = [activations[-1] - y]  # Compute the initial delta (error) for the output layer

        for i in reversed(range(len(self.weights) - 1)):
            # Compute delta for each layer
            delta = np.dot(deltas[-1], self.weights[i + 1].T) * self.relu_derivative(pre_activations[i])
            deltas.append(delta)  # Append delta to the list

        deltas.reverse()  # Reverse the list of deltas to match layer order

        for i in range(len(self.weights)):
            # Update weights and biases using the computed deltas
            self.weights[i] -= self.learning_rate * np.dot(activations[i].T, deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m

    def compute_loss(self, y, y_hat):
        # Compute mean squared error loss
        return np.mean(np.square(y - y_hat))

    def train(self, X, y, epochs=10000, batch_size=32):
        # Train the neural network using mini-batch gradient descent
        m = X.shape[0]  # Number of training examples
        self.losses = []  # List to store loss values

        for epoch in range(epochs):
            # Shuffle the dataset at the beginning of each epoch
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, m, batch_size):
                # Create mini-batches
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Perform forward and backward propagation
                activations, pre_activations = self.forward_propagation(X_batch)
                self.backward_propagation(X_batch, y_batch, activations, pre_activations)

            # Compute loss for the entire dataset
            y_hat = self.forward_propagation(X)[0][-1]
            loss = self.compute_loss(y, y_hat)
            self.losses.append(loss)  # Store the loss value

            if epoch % 1000 == 0:
                # Print loss every 1000 epochs
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        # Predict the output for the given input X
        return self.forward_propagation(X)[0][-1]

# Example usage:
if __name__ == "__main__":
    # Sample data (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Define the architecture (2 input neurons, 2 hidden layers with 4 neurons each, and 1 output neuron)
    layers = [2, 4, 4, 1]
    ann = AdvancedANN(layers, learning_rate=0.01)  # Initialize the neural network with the defined architecture
    ann.train(X, y, epochs=10000, batch_size=2)  # Train the network

    # Make predictions
    predictions = ann.predict(X)
    print("Predictions:")
    print(predictions)
