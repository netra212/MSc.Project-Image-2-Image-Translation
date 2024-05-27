"""
Today, I will practice on the same datasets i.e.., Insurance datasets that i have previously used in the 01_neuron.txt. just an binary classification. 
In the same datasets, 
I can have the multiple inputs like
Age, Education, Income, Saving. 
"""

import tensorflow as tf
from tensorflow import keras 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 

# separating an datasets. 
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("X_train is Two dimesional :")
print("shape of X_train", X_train.shape)
print("shape of y_train", y_train.shape)
print("X_test is Two dimesional :")
print("shape of X_test", X_test.shape)
print("shape of y_test", y_test.shape)

# Looking the sample datasets. 
plt.imshow(X_train[0])
# plt.show()

# since, the we just have checked the shape of the X_train <-- input features, contain the two-dimensional matrix, but we have to pass the one dimensional matrix as input so we have to convert the two-dimensional input arrays into single dimension.
# In order to convert the two-dimensional input -> one-dimensional input, we have to use the panda's reshape function, which takes two parameters as input. 
# shape of X_train (60000, 28, 28)
# shape of X_test (10000, 28, 28)
# 60000 -> represent the first dimension. 
# 28,28 -> represent the 2nd & 3rd dimension are each individual image. 
# Below is the code:

# Before converting the two-dimensional arrays into an one dimensional arrays, Let's just scaled an datasets to a certain range, for that we have to divide the whole arrays with 255 since the value is ranges from 0 to 255 so. we have divided with the 255. we did this scaling just to improve the accuracy.
# Note here:- We don't scale the y_train and y_test data, we just do the scaling for the X_train and X_test datasets. 
 
X_train = X_train / 255 
X_test = X_test / 255

X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)

print("Shape of X_train flattened is : ", X_train_flattened.shape)
print("Shape of X_test flattened is : ", X_test_flattened.shape)

# Keras.Sequential()
# Sequential() --> means I am having an stack of layers in my neural networks. 
# since it is an stack, so it will accept an every layers as one element. 
# Dense -> Means All the neurons in one layer is connected with every other neurons in other layers. 
# output -> 0 to 9 since the image is ranges from 0,1,2,3 up to 9.
# input_shape -> 784 represent the total number of the inputs. 
# activation -> activation function. 

model = keras.Sequential([ 
    keras.layers.Dense(10, input_shape=(784,),activation='sigmoid')
])
model.compile(
    optimizer='adam', # optimizer allows you to train efficiently and helps to reach global optima.
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'] # metrics is accuracy while compiling.
)
model.fit(X_train_flattened,y_train, epochs=5) # training actually happens here... we passed train datasets here.

# Let's check the accuracy on the test datasets. 
accuracy_test_datasets = model.evaluate(X_test_flattened, y_test)
print("Checking the accuracy on the test datasets: ")
print(accuracy_test_datasets)

# Let's predict 
print("Below code should print the 7 since the first value of the X_test is 7 so : ")
plt.imshow(X_test[0])
# plt.show()
y_predicted = model.predict(X_test_flattened)
print("\nIt will display the 10 scores for the value of 7 because the first value is the 7 so :", y_predicted[0])
# since above code is displaying 10 values so we have calculate the maximum value and needs to print the index of that max value. 
print("Below code should predict the 7 since the value at the position of X_test[0] is 7 :")
print(np.argmax(y_predicted[0]))

# Let's print the confusion matrix is:
# Need to convert the y_predicted into the whole number.
print("Values in the y_test: ", y_test[:5])
y_predicted_labels = [np.argmax(i) for i in y_predicted]
print("Values in the y_predicted labels : ", y_predicted_labels[:5])

confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
print("Printing the first 5 values : ", confusion_matrix)


# Displaying the confusion matrix in the visual format with the help of seaborn library. 
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()


"""
********************************
Let's Add me one Hidden Layer:
********************************
"""
# since, we are adding the hidden layers, so we have to specificy the number of neurons like this ... 
number_neurons = 100

model = keras.Sequential([ 
    keras.layers.Dense(number_neurons, input_shape=(784,),activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
model.compile(
    optimizer='adam', # optimizer allows you to train efficiently and helps to reach global optima.
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'] # metrics is accuracy while compiling.
)
model.fit(X_train_flattened,y_train, epochs=5) # training actually happens here... we passed train datasets here.

# Let's check the accuracy on the test datasets. 
accuracy_test_datasets = model.evaluate(X_test_flattened, y_test)
print("Checking the accuracy on the test datasets: ")
print(accuracy_test_datasets)

# Let's predict 
print("Below code should print the 7 since the first value of the X_test is 7 so : ")
plt.imshow(X_test[0])
# plt.show()
y_predicted = model.predict(X_test_flattened)
print("\nIt will display the 10 scores for the value of 7 because the first value is the 7 so :", y_predicted[0])
# since above code is displaying 10 values so we have calculate the maximum value and needs to print the index of that max value. 
print("Below code should predict the 7 since the value at the position of X_test[0] is 7 :")
print(np.argmax(y_predicted[0]))

# Let's print the confusion matrix is:
# Need to convert the y_predicted into the whole number.
print("Values in the y_test: ", y_test[:5])
y_predicted_labels = [np.argmax(i) for i in y_predicted]
print("Values in the y_predicted labels : ", y_predicted_labels[:5])

confusion_matrix = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
print("Printing the first 5 values : ", confusion_matrix)


# Displaying the confusion matrix in the visual format with the help of seaborn library. 
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

# Using the built keras method for Flattening.

model = keras.Sequential([ 
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(number_neurons,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
model.compile(
    optimizer='adam', # optimizer allows you to train efficiently and helps to reach global optima.
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'] # metrics is accuracy while compiling.
)
model.fit(X_train_flattened,y_train, epochs=5) # training actually happens here... we passed train datasets here.
