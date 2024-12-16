import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class MultiLayerNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        
        np.random.seed(42)
        # Инициализация весов методом Xavier/Glorot
        w1 = np.random.uniform(
            -np.sqrt(6.0 / (layer_sizes[0] + layer_sizes[1])),
            np.sqrt(6.0 / (layer_sizes[0] + layer_sizes[1])),
            (layer_sizes[0], layer_sizes[1])
        ).astype(np.float64)
        
        weights = [w1]
        biases = [np.zeros((1, layer_sizes[1]), dtype=np.float64)]
        
        for i in range(1, len(layer_sizes) - 1):
            w = np.random.uniform(
                -np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1])),
                np.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1])),
                (layer_sizes[i], layer_sizes[i + 1])
            ).astype(np.float64)
            weights.append(w)
            biases.append(np.zeros((1, layer_sizes[i + 1]), dtype=np.float64))
        
        self.weights = weights
        self.biases = biases
    
    def forward_propagation(self, x):
        self.activations = [x]
        self.u = []
        
        for i in range(len(self.weights)):
            u = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.u.append(u)
            activation = sigmoid(u)
            self.activations.append(activation)
        
        return self.activations[-1]
    
    def backward_propagation(self, x, y, learning_rate):
        error = y - self.activations[-1]
        delta = error * sigmoid_derivative(self.u[-1])
        deltas = [delta]
        
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(self.u[i-1])
            deltas.append(delta)
        
        deltas.reverse()
        
        for i in range(len(self.weights)):
            delta_w = learning_rate * np.dot(self.activations[i].T, deltas[i])
            delta_b = learning_rate * np.sum(deltas[i], axis=0, keepdims=True)
            
            # L2 регуляризация
            delta_w -= 0.0001 * learning_rate * self.weights[i]
            
            self.weights[i] += delta_w
            self.biases[i] += delta_b
    
    def train(self, X, y, epochs, learning_rate=0.01):
        history = []
        best_error = float('inf')
        best_weights = None
        best_biases = None
        patience = 50
        no_improve = 0
        
        for epoch in range(epochs):
            # Перемешиваем данные
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            
            output = self.forward_propagation(X)
            error = np.mean(np.square(y - output))
            history.append(error)
            
            if error < best_error:
                best_error = error
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
            
            # Адаптивная скорость обучения
            current_lr = learning_rate / (1 + epoch * 0.0001)
            self.backward_propagation(X, y, current_lr)
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Error: {error:.6f}')
        
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases
        
        return history

print("\nЗадание для 3lab_data.csv:")
data_3lab = pd.read_csv('3lab_data.csv')
x_3lab = data_3lab[['x1', 'x2', 'x3']].values
y_3lab = data_3lab[['y1', 'y2']].values

x_min = x_3lab.min(axis=0)
x_max = x_3lab.max(axis=0)
x_3lab = (x_3lab - x_min) / (x_max - x_min)

y_min = y_3lab.min(axis=0)
y_max = y_3lab.max(axis=0)
y_3lab = (y_3lab - y_min) / (y_max - y_min)

train_size_3lab = int(0.8 * len(x_3lab))
x_train = x_3lab[:train_size_3lab]
y_train = y_3lab[:train_size_3lab]
x_test = x_3lab[train_size_3lab:]
y_test = y_3lab[train_size_3lab:]

mlp_3lab = MultiLayerNetwork([3, 8, 6, 2])  # Увеличили количество слоев и нейронов
history_3lab = mlp_3lab.train(x_train, y_train, epochs=15000, learning_rate=0.001)

y_pred_3lab = mlp_3lab.forward_propagation(x_test)

y_pred_original = y_pred_3lab * (y_max - y_min) + y_min
y_test_original = y_test * (y_max - y_min) + y_min

mse_3lab = np.mean(np.square(y_test_original - y_pred_original), axis=0)
print(f'MSE для каждого выхода: {mse_3lab}')
print(f'Общая MSE: {np.mean(mse_3lab)}')

print("\nВеса нейронной сети:")
for i in range(len(mlp_3lab.weights)):
    print(f"\nСлой {i+1} (входной -> скрытый):")
    print(f"Размерность весов: {mlp_3lab.weights[i].shape}")
    print("Матрица весов:")
    print(mlp_3lab.weights[i])
    print("\nСмещения:")
    print(mlp_3lab.biases[i])

print("\nСравнение предсказаний с реальными значениями (в исходном масштабе):")
print("\nФормат: [y1_реальное, y2_реальное] -> [y1_предсказанное, y2_предсказанное]")
for i in range(len(y_test_original)):
    print(f"Реальные значения: [{y_test_original[i][0]:8.4f}, {y_test_original[i][1]:8.4f}] -> "
          f"Предсказанные: [{y_pred_original[i][0]:8.4f}, {y_pred_original[i][1]:8.4f}]")

mae = np.mean(np.abs(y_test_original - y_pred_original), axis=0)
print("\nСредняя абсолютная ошибка:")
print(f"Выход 1: {mae[0]:.4f}")
print(f"Выход 2: {mae[1]:.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.scatter(range(len(y_test_original)), y_test_original[:, 0], label='Реальные значения', color='blue')
plt.scatter(range(len(y_pred_original)), y_pred_original[:, 0], label='Предсказанные значения', color='red')
plt.title('Сравнение для первого выхода (y1)')
plt.xlabel('Номер примера')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(range(len(y_test_original)), y_test_original[:, 1], label='Реальные значения', color='blue')
plt.scatter(range(len(y_pred_original)), y_pred_original[:, 1], label='Предсказанные значения', color='red')
plt.title('Сравнение для второго выхода (y2)')
plt.xlabel('Номер примера')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()