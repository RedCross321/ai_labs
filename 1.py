import numpy as np
import pandas as pd
import random
from PIL import Image
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.uniform(-0.01, 0.01, num_inputs)
        self.bias = random.uniform(-0.01, 0.01)

    def calculate(self, inputs):
        result = np.dot(inputs, self.weights) + self.bias
        if np.isinf(result) or np.isnan(result): # часто выпадало в ошибку, пришлось начать проверять)
            raise ValueError("NaN или бесконечность")
        return result

class NeuralNetwork:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def predict(self, x):
        return np.array([neuron.calculate(x) for neuron in self.neurons])

    def predict_image(self, pixels):
        return np.array([self.predict(pixel[:2]) for pixel in pixels])

    def fit_1(self, x, y, learning_rate=0.0001, target_error=0.001, epochs=1000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in zip(x, y):
                for neuron in self.neurons:
                    prediction = neuron.calculate(inputs)
                    error = target - prediction
                    total_error += abs(error)

                    for i in range(len(neuron.weights)):
                        neuron.weights[i] += learning_rate * error * inputs[i]
                    neuron.bias += learning_rate * error

            if total_error <= target_error:
                print(f"Сеть сошлась на эпохе {epoch + 1} с ошибкой {total_error}, обучение остановлено.")
                return


    def fit_2(self, x, y, learning_rate=0.00001, target_error=0.001, epochs=1000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in zip(x, y):
                for neuron in self.neurons:
                    prediction = neuron.calculate(inputs)
                    error = target - prediction
                    total_error += 0.5 * (error ** 2)

                    for i in range(len(neuron.weights)):
                        gradient = -error * inputs[i]
                        neuron.weights[i] -= learning_rate * gradient
                    neuron.bias -= learning_rate * (-error)

            if total_error <= target_error:
                print(f"Сеть сошлась на эпохе {epoch + 1} с ошибкой {total_error}, обучение остановлено.")
                return

    def fit_3(self, x, y, learning_rate=0.0001, target_error=0.001, epochs=1000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, targets in zip(x, y):
                for idx, neuron in enumerate(self.neurons):
                    prediction = neuron.calculate(inputs)
                    error = targets[idx] - prediction
                    total_error += 0.5 * (error ** 2)

                    for i in range(len(neuron.weights)):
                        gradient = -error * inputs[i]
                        neuron.weights[i] -= learning_rate * gradient
                    neuron.bias -= learning_rate * (-error)

            if total_error <= target_error:
                print(f"Сеть сошлась на эпохе {epoch + 1} с ошибкой {total_error}, обучение остановлено.")
                return

# задание 1
data_1 = pd.read_csv('2lab.csv')
x_1 = data_1[['x1', 'x2']].values
y_1 = data_1['y'].values

x_1 = (x_1 - np.mean(x_1, axis=0)) / np.std(x_1, axis=0)

train_size = int(0.8 * len(x_1))
x_train, x_test = x_1[:train_size], x_1[train_size:]
y_train, y_test = y_1[:train_size], y_1[train_size:]

nn1 = NeuralNetwork(num_neurons=1, num_inputs=2)
# nn2 = NeuralNetwork(num_neurons=1, num_inputs=2)

# print("Обучение по первому варианту:")
nn1.fit_1(x_train, y_train, learning_rate=0.0001, target_error=0.001, epochs=1000)

# for idx, neuron in enumerate(nn1.neurons):
#     print(f"Нейрон {idx + 1}: веса = {neuron.weights}, смещение = {neuron.bias}")

# print("Обучение по второму варианту:")
# nn2.fit_2(x_train, y_train, learning_rate=0.00001, target_error=0.001, epochs=1000)
# for idx, neuron in enumerate(nn2.neurons):
#     print(f"Нейрон {idx + 1}: веса = {neuron.weights}, смещение = {neuron.bias}")

# def mean_squared_error(y_true, y_pred):
#     return np.mean((y_true - y_pred) ** 2)

# y_pred_1 = [nn1.predict(x)[0] for x in x_test]
# y_pred_2 = [nn2.predict(x)[0] for x in x_test]

# mse_1 = mean_squared_error(y_test, y_pred_1)
# mse_2 = mean_squared_error(y_test, y_pred_2)

# print(f"Среднеквадратичная ошибка для первого варианта обучения: {mse_1}")
# print(f"Среднеквадратичная ошибка для второго варианта обучения: {mse_2}")

# # задание 2
# data_2 = pd.read_csv('2lab_data.csv')
# x_2 = data_2[['x1', 'x2', 'x3', 'x4', 'x5', 'x6']].values
# y_2 = data_2[['y1', 'y2', 'y3']].values

# x_2 = (x_2 - np.mean(x_2, axis=0)) / np.std(x_2, axis=0)

# train_size_2 = int(0.8 * len(x_2))
# x_train_2, x_test_2 = x_2[:train_size_2], x_2[train_size_2:]
# y_train_2, y_test_2 = y_2[:train_size_2], y_2[train_size_2:]

# nn = NeuralNetwork(num_neurons=3, num_inputs=6)
# print("Обучение во втором задании:")
# nn.fit_3(x_train_2, y_train_2, learning_rate=0.0001, target_error=0.001, epochs=1000)
# for idx, neuron in enumerate(nn.neurons):
#     print(f"Нейрон {idx + 1}: веса = {neuron.weights}, смещение = {neuron.bias}")

# y_pred_2 = np.array([nn.predict(x) for x in x_test_2])

# mse_2 = [mean_squared_error(y_test_2[:, i], y_pred_2[:, i]) for i in range(y_2.shape[1])]

# for i, error in enumerate(mse_2):
#     print(f"Среднеквадратичная ошибка для выхода {i + 1}: {error}")

# задание 3
def process_image(image_path, nn):
    a = np.asarray(Image.open(image_path))
    mas = a.reshape(-1, 3) # хз почему не работает с (1, 3). наковырял на -1
    res = []
    res = nn.predict_image(mas)
    our_array = np.array(res).reshape(a.shape[0], a.shape[1])
    return our_array * 255

processed_image = process_image('pic.jpg', nn1)
plt.figure(figsize=(15., 10.))
plt.imshow(processed_image, cmap='gray')
plt.show()