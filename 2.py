import math
import random
from re import X
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import csv


class Neurone:
    def __init__(self, n):
        self.number = n
        self.weight = np.random.uniform(0.001,0.2,n)
        self.bias = np.random.uniform(0.001,0.2)
        self.somma = 0

    def prevedere(self, x):
        self.somma = 0
        
        for i in range(self.number):
            self.somma += x[i] * self.weight[i] + self.bias
        return self.somma


class NeuralNetwork(Neurone):
    def __init__(self, number, input):
        self.nero = []
        for _ in range(number):
            self.nero.append(Neurone(input))

    def predict(self, x):
        predictions = []
        for neuron in self.nero:
            predictions.append(neuron.prevedere(x))
        return np.array(predictions)

    def fit_1(self, data, x, target_error, learning_rate=0.01):
        data_list = list(data)
        for _ in range(x):
            total_error = 0
            for inputs, target in data_list:
                for neuron in self.nero:
                    prediction = neuron.prevedere(inputs)
                    error = target - prediction
                    total_error += error ** 2
                    
                    for i in range(len(neuron.weight)):
                        neuron.weight[i] += learning_rate * error * inputs[i]

            mse = total_error / len(data_list)
            if mse <= target_error:
                return

    def fit_2(self, data, x, target_error, learning_rate):
        data_list = list(data)
        for _ in range(x):
            total_error = 0
            for inputs, target in data_list:
                for neuron in self.nero:
                    prediction = neuron.prevedere(inputs)
                    error = target - prediction
                    total_error += error * error
                    for i in range(len(neuron.weight)):
                        gradient = -2 * error * inputs[i]
                        neuron.weight[i] -= gradient

                    bias_gradient = -2 * error
                    neuron.bias -= learning_rate * bias_gradient

            if len(data_list) > 0:
                mse = total_error / len(data_list)
                if mse <= target_error:
                    return

    def fit_3(self, data, learning_rate, target_error, epochs):
        data_list = list(data)
        for _ in range(epochs):
            total_error = 0
            for inputs, targets in data_list:
                for idx, neuron in enumerate(self.nero):
                    prediction = neuron.prevedere(inputs)
                    error = targets[idx] - prediction
                    total_error += error ** 2
                    
                    for i in range(len(neuron.weight)):
                        neuron.weight[i] += learning_rate * error * inputs[i]
                    neuron.bias += learning_rate * error

            mse = total_error / (len(data_list) * len(self.nero))
            if mse <= target_error:
                return

    def fit_image(self, data, x, target_error, learning_rate=0.01):
        data_list = list(data)
        learning_rate = 0.00001 
        
        for _ in range(x):
            total_error = 0
            for inputs, target in data_list:
                target = target / 255.0
                
                for neuron in self.nero:
                    prediction = neuron.prevedere(inputs)
                    error = target - prediction
                    total_error += min(100, error * error)
                    
                    inputs_norm = np.array(inputs) / 255.0
                    
                    for i in range(len(neuron.weight)):
                        update = learning_rate * error * inputs_norm[i]
                        update = max(-0.1, min(0.1, update))
                        neuron.weight[i] += update
                    
                    bias_update = learning_rate * error
                    bias_update = max(-0.1, min(0.1, bias_update))
                    neuron.bias += bias_update

            mse = total_error / len(data_list)
            if mse <= target_error:
                return

    def predict_image(self, image_array):
        predictions = []
        for pixel in image_array:
            pred = self.predict(pixel.flatten())[0]
            predictions.append(pred)
        return np.array(predictions)

def squared_error(y_true, y_pred):
    for i in range(len(y_true)):
        mas = 0.5 * ((y_true[i] - y_pred[i]) ** 2)
    return np.mean(mas)
data = pd.read_csv('test.csv')


# x_1 = data[['x1', 'x2']].values
# y_1 = data['y'].values

# train_size = int(0.8 * len(x_1))
# inputs = zip(x_1[:train_size], y_1[:train_size])

# # Neur_n = int(input("Enter number of neurone -> "))
# # num_input = int(input("Enter number of input -> "))


# run1 = NeuralNetwork(1, 2)
# run2 = NeuralNetwork(1, 2)

# learning_rate = 0.0001  
# target_error = 0.0001 
# epochs = 1000
# run1.fit_1(inputs, epochs, target_error, learning_rate)
# run2.fit_2(inputs, epochs, target_error, learning_rate)

# for id, neuron in enumerate(run1.nero):
#     print(f"Нейрон {id + 1}: веса = {neuron.weight}, смещение = {neuron.bias}")

# for id, neuron in enumerate(run2.nero):
#     print(f"Нейрон {id + 1}: веса = {neuron.weight}, смещение = {neuron.bias}")

# y_pred_1 = [run1.predict(x)[0] for x in x_1[train_size:]]
# y_pred_2 = [run2.predict(x)[0] * (-1) for x in x_1[train_size:]]
# # for inputs, target1, target2 in zip(y_1[train_size:], y_pred_1, y_pred_2):  
# #    print(inputs, target1, target2)
# mse_1 = squared_error(y_1[train_size:], y_pred_1)
# mse_2 = squared_error(y_1[train_size:], y_pred_2)
# print(f"Среднеквадратичная ошибка для первого варианта обучения: {mse_1}")
# print(f"Среднеквадратичная ошибка для второго варианта обучения: {mse_2}")

# data_2 = pd.read_csv('2lab_data.csv')
# x_2 = data_2[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
# y_2 = data_2[['y1', 'y2', 'y3']].values

# train_size_2 = int(0.8 * len(x_2))

# inputs = zip(x_2[:train_size_2], y_2[:train_size_2])

# run3 = NeuralNetwork(3, 6)
# print("Обучение во втором задании:")

# run3.fit_3(inputs, learning_rate=0.0001, target_error=0.001, epochs=1000)
# for id, neuron in enumerate(run3.nero):
#     print(f"Нейрон {id + 1}: веса = {neuron.weight}, смещение = {neuron.bias}")

# y_pred_2 = np.array([run3.predict(x) for x in x_2[train_size_2:]])

# mse_2 = [squared_error(y_2[train_size_2:, i], y_pred_2[:, i]) for i in range(y_2.shape[1])]

# for i, error in enumerate(mse_2):
#     print(f"Среднеквадратичная ошибка для выхода {i + 1}: {error}")

data_3 = pd.read_csv('2lab_data_3_train.csv')
x_3 = data_3[['R', 'G', 'B']].values  
y_3 = data_3['y'].values  

train_size_3 = int(0.8 * len(x_3))
inputs_3 = zip(x_3[:train_size_3], y_3[:train_size_3])


run_image = NeuralNetwork(1, 3)  
run_image.fit_image(inputs_3, x=1000, target_error=0.0001, learning_rate=0.0001)


img = np.asarray(Image.open('pic.jpg'))
img_reshaped = img.reshape(-1, 3)  

predictions = run_image.predict_image(img_reshaped)
output_image = predictions.reshape(img.shape[0], img.shape[1])  

plt.figure(figsize=(15, 10))
plt.imshow(output_image, cmap='gray')
plt.title('Преобразованное изображение')
plt.axis('off')
plt.show()