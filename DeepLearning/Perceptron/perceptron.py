# Importing Libraries. 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

# reading datasets. 
df = pd.read_csv("/Users/netrakc/Desktop/MSc.Project-Image-2-Image-Translation/DeepLearning/placement.csv")

# shape. 
print("\nshape of the datasets: ", df.shape)
print(df.head(5))
print("\n")

# plotting the datasets. 
# print("\nscatter plot of the datasets:")
# sns.scatterplot(x=df['cgpa'],y=df['resume_score'],hue=df['placed'])
# plt.title('Scatter Plot of CGPA vs Resume Score')
# plt.xlabel('CGPA')
# plt.ylabel('Resume Score')
# plt.show()

# splitting of the datasets:
print("\nSplitting of the datasets:")
X = df.iloc[:,0:2]
y = df.iloc[:,-1]

print("shape of X: ", X.shape)
print("Shape of y: ", y.shape)
print("\n")

class Perceptron:

    def __init__(self, learning_rate=0.01,n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None # Goal is to calculate the value of the weights. 
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # first, weights are initialized randomly. 
        weights = np.zeros(n_features)
        # bias is initialized to zero. 
        self.bias = 0

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                z_linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation_function(z_linear_output)

                # Update the Perceptron.
                update_perceptron = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update_perceptron*x_i
                self.bias += update_perceptron

    def predict(self, X):
        z_linear_output = np.dot(X,self.weights) + self.bias
        y_predicted = self._activation_function(z_linear_output)
        return y_predicted
    
    def _activation_function(self,z_linear_output):
        return np.where(z_linear_output >= 0, 1, 0)

perceptron = Perceptron(learning_rate=0.01, n_iter=1000)

# Training the Perceptron Networks. 
print("\nTraining the perceptron networks:")
perceptron.fit(X,y)

# Prediction 
prediction = perceptron.predict(X)
print(prediction)






