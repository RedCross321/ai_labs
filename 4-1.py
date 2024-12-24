import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X = np.arange(-20, 20.1, 0.1)
y = X

X = X.reshape(-1, 1)

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Задание 1 - Линейная функция
model = Sequential([
    Dense(4, activation='linear', input_shape=(1,)),
    Dense(1, activation='linear', input_shape=(1,))
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

test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f'\nТочность на тестовой выборке (MSE): {test_loss[0]:.12f}')
print(f'MAE на тестовой выборке: {test_loss[1]:.12f}')

y_pred = model.predict(X_test)

plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='blue', label='Реальные значения', alpha=0.5)
plt.scatter(X_test, y_pred, color='red', label='Предсказания', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('График реальной функции и предсказаний нейронной сети')
plt.legend()
plt.grid(True)
plt.show()

# plt.figure(figsize=(12, 6))
# plt.plot(history.history['loss'], label='Ошибка обучения')
# plt.plot(history.history['val_loss'], label='Ошибка валидации')
# plt.xlabel('Эпоха')
# plt.ylabel('Ошибка (MSE)')
# plt.title('График ошибки обучения')
# plt.legend()
# plt.grid(True)
# plt.show()

weights = model.get_weights()
print("\nВеса модели:")
print(f"Коэффициент наклона (вес): {weights[0][0][0]:.12f}")
print(f"Смещение (bias): {weights[1][0]:.12f}")

max_error = np.max(np.abs(y_pred - y_test.reshape(-1, 1)))
print(f"\nМаксимальная абсолютная ошибка: {max_error:.12f}")