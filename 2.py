import math
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

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
        for _ in range(number):
            self.nero = [Neurone(input)]

    def predict(self, x):
        return np.array([neuron.prevedere(x) for neuron in self.nero])

    def fit_1(self, data, x, target_error):
        for i in range(x):
            total_error = 0
            num_ind = 0
            for inputs, target in data:
                for neuron in self.nero:
                    prediction = neuron.prevedere(inputs)
                    error = (target - prediction)
                    total_error += abs(error)
                    # print(total_error)
                #print(inputs)
                num_ind += 1
                #input()
            total_error = total_error/num_ind
            for i in range(len(neuron.weight)):
                neuron.weight[i] += error / sum(inputs[i] for i in range(0, int(len(inputs))))
                # print(total_error)
            print(i)
                # if total_error < target_error:
                # print(total_error, '<', target_error)
                    # return
            
    def fit_2(self, data, x, target_error, learning_rate):
        for _ in range(x):
            total_error = 0
            for inputs, target in data:
                for neuron in self.nero:
                    prediction = neuron.prevedere(inputs)
                    error = target - prediction * len(neuron.weight)
                    total_error += 0.5 * (error ** 2)

                    for i in range(len(neuron.weight)):
                        neuron.weight[i] -= learning_rate * -error * inputs[i]
                    

            if total_error <= target_error:
                return


def impl(a):
    return 0 if a > 0 else 1

def cos(a):
    return math.cos(a)

def chet(a):
    return a

def squared_error(y_true, y_pred):
    for i in range(len(y_true)):
        mas = 0.5 * ((y_true[i] - y_pred[i]) ** 2)
    return np.mean(mas)
data = pd.read_csv('2lab.csv')


x_1 = data[['x1', 'x2']].values
y_1 = data['y'].values

train_size = int(0.8 * len(x_1))
inputs = zip(x_1[:train_size], y_1[:train_size])

Neur_n = int(input("Enter number of neurone -> "))
num_input = int(input("Enter number of input -> "))


run1 = NeuralNetwork(Neur_n, num_input)
run2 = NeuralNetwork(Neur_n, num_input)

learning_rate = 0.0001
target_error = 0.00001
epochs = 1000
run1.fit_1(inputs, epochs, target_error)
run2.fit_2(inputs, epochs, target_error, learning_rate)

for id, neuron in enumerate(run1.nero):
    print(f"Нейрон {id + 1}: веса = {neuron.weight}, смещение = {neuron.bias}")

for id, neuron in enumerate(run2.nero):
    print(f"Нейрон {id + 1}: веса = {neuron.weight}, смещение = {neuron.bias}")

y_pred_1 = [run1.predict(x)[0] for x in x_1[train_size:]]
y_pred_2 = [run2.predict(x)[0] for x in x_1[train_size:]]
for inputs, target in zip(y_1[train_size:], y_pred_1):  
    print(inputs, target)
mse_1 = squared_error(y_1[train_size:], y_pred_1)
mse_2 = squared_error(y_1[train_size:], y_pred_2)
print(f"Среднеквадратичная ошибка для первого варианта обучения: {mse_1}")
print(f"Среднеквадратичная ошибка для второго варианта обучения: {mse_2}")











# def relu(a):
#     if a > 0 :
#         return a
#     else:
#         return 0

    
# X = [[] for i in range(100)]
# I = 0

# for i in X:
#     I+=1
#     i.append(math.sin(math.pi * 2 + math.pi/4 + I*I) + math.pi )
#     i.append(I)
#     i.append(I * math.sin(math.pi*2+math.pi/2 + I))

# dd.set_w([1, 1, 1])

# for i in range(100):
#     print(X[i],"  ",dd.prevedere(X[i],relu))

# for _ in range(10):
#     inp = [random.randint(0,1) for _ in range(x)]
#     print(inp,"",len(inp),dd.prevedere(inp, impl))