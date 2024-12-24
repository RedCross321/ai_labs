import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X = np.arange(-20, 20.1, 0.1)
y = np.abs(X)

X = X.reshape(-1, 1)

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = Sequential([
    Dense(16, activation='relu', input_shape=(1,)),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='relu')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=4,
    validation_split=0.2,
    verbose=1
)

test_loss = model.evaluate(X, y, verbose=0)
print(f'\nТочность на тестовой выборке (MSE): {test_loss[0]:.12f}')
print(f'MAE на тестовой выборке: {test_loss[1]:.12f}')

y_pred = model.predict(X)

plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Реальные значения', alpha=0.5)
plt.scatter(X, y_pred, color='red', label='Предсказания', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('График функции |x| и предсказаний нейронной сети')
plt.legend()
plt.grid(True)
plt.show()

max_error = np.max(np.abs(y_pred - y.reshape(-1, 1)))
print(f"\nМаксимальная абсолютная ошибка: {max_error:.12f}")