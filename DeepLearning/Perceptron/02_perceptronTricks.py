from sklearn.datasets import make_classification
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import random

X,y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=41, hypercube=False, class_sep=10)


# Defining the perceptron.
def perceptron(X,y):

    # Adding one to the first.
    X = np.ones(X,0,1, axis=1)
    
    # creating an weights which shapes is equal to the X. 
    weights = np.ones(X.shape[1])

    # Initializing the learning rate. 
    lr = 0.01 

    # Deciding the number of the epochs. 
    epochs = 1000

    # Looping through the students.
    for i in range(epochs):

        # out of the 100 student, selecting the random student, j <- represent the random student. 
        j = np.random.randint(0,100)

        # making an prediction.
        y_hat = step(np.dot(X[j], weights))

        # calculating weights update. 
        weights = weights + lr * (y[j] - y_hat) * X[j]
    
    return weights[0], weights[1:]
    # weights[0] <-- intercept value.
    # weights[1] <-- coefficent value. 

def step(z):
    return 1 if z > 0 else 0

intercept_, coef_ = perceptron(X,y)

print("value of coef_ : ", coef_)
print("value of intercept_ : ", intercept_)



























