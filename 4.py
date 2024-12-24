import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# X = np.arange(-20, 20.1, 0.1)
# y = X

# X = X.reshape(-1, 1)

# split_idx = int(len(X) * 0.8)
# X_train, X_test = X[:split_idx], X[split_idx:]
# y_train, y_test = y[:split_idx], y[split_idx:]

# # Задание 1 - Линейная функция
# model = Sequential([
#     Dense(4, activation='linear', input_shape=(1,)),
#     Dense(1, activation='linear', input_shape=(1,))
# ])

# optimizer = Adam(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# history = model.fit(
#     X_train, y_train,
#     epochs=50,
#     batch_size=4,
#     validation_split=0.2,
#     verbose=1
# )

# test_loss = model.evaluate(X_test, y_test, verbose=0)
# print(f'\nТочность на тестовой выборке (MSE): {test_loss[0]:.12f}')
# print(f'MAE на тестовой выборке: {test_loss[1]:.12f}')

# y_pred = model.predict(X_test)

# plt.figure(figsize=(12, 6))
# plt.scatter(X_test, y_test, color='blue', label='Реальные значения', alpha=0.5)
# plt.scatter(X_test, y_pred, color='red', label='Предсказания', alpha=0.5)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('График реальной функции и предсказаний нейронной сети')
# plt.legend()
# plt.grid(True)
# plt.show()

# # plt.figure(figsize=(12, 6))
# # plt.plot(history.history['loss'], label='Ошибка обучения')
# # plt.plot(history.history['val_loss'], label='Ошибка валидации')
# # plt.xlabel('Эпоха')
# # plt.ylabel('Ошибка (MSE)')
# # plt.title('График ошибки обучения')
# # plt.legend()
# # plt.grid(True)
# # plt.show()

# weights = model.get_weights()
# print("\nВеса модели:")
# print(f"Коэффициент наклона (вес): {weights[0][0][0]:.12f}")
# print(f"Смещение (bias): {weights[1][0]:.12f}")

# max_error = np.max(np.abs(y_pred - y_test.reshape(-1, 1)))
# print(f"\nМаксимальная абсолютная ошибка: {max_error:.12f}")


# Задание 2 - Нелинейная функция
# X = np.arange(-20, 20.1, 0.1)
# y = np.abs(X)

# X = X.reshape(-1, 1)

# split_idx = int(len(X) * 0.8)
# X_train, X_test = X[:split_idx], X[split_idx:]
# y_train, y_test = y[:split_idx], y[split_idx:]

# model = Sequential([
#     Dense(16, activation='relu', input_shape=(1,)),
#     Dense(8, activation='relu'),
#     Dense(4, activation='relu'),
#     Dense(1, activation='relu')
# ])

# optimizer = Adam(learning_rate=0.001)
# model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# history = model.fit(
#     X_train, y_train,
#     epochs=50,
#     batch_size=4,
#     validation_split=0.2,
#     verbose=1
# )

# test_loss = model.evaluate(X, y, verbose=0)
# print(f'\nТочность на тестовой выборке (MSE): {test_loss[0]:.12f}')
# print(f'MAE на тестовой выборке: {test_loss[1]:.12f}')

# y_pred = model.predict(X)

# plt.figure(figsize=(12, 6))
# plt.scatter(X, y, color='blue', label='Реальные значения', alpha=0.5)
# plt.scatter(X, y_pred, color='red', label='Предсказания', alpha=0.5)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('График функции |x| и предсказаний нейронной сети')
# plt.legend()
# plt.grid(True)
# plt.show()

# max_error = np.max(np.abs(y_pred - y.reshape(-1, 1)))
# print(f"\nМаксимальная абсолютная ошибка: {max_error:.12f}")

# Задание 3 - Окружность
# radius = 20
# X_edges = np.linspace(-radius, -radius, 200)
# X_edges = np.append(X_edges, np.linspace(radius, radius, 200))
# X_middle = np.linspace(-radius, radius, 800)
# X = np.concatenate([X_edges, X_middle])
# X = np.sort(X)

# valid_mask = np.abs(X) <= radius
# X = X[valid_mask]

# Y_up = np.sqrt(radius**2 - X**2)
# Y_down = -np.sqrt(radius**2 - X**2)

# X = np.concatenate([X, X])
# Y = np.concatenate([Y_up, Y_down])
# signs = np.concatenate([np.ones_like(Y_up), -np.ones_like(Y_down)])

# X_norm = X / radius
# Y_norm = Y / radius

# X_features = np.column_stack([X_norm, signs])

# indices = np.random.permutation(len(X))
# X_features = X_features[indices]
# Y_norm = Y_norm[indices]

# model = Sequential([
#     Dense(256, activation='tanh', input_shape=(2,)),
#     Dense(256, activation='tanh'),
#     Dense(256, activation='tanh'),
#     Dense(128, activation='tanh'),
#     Dense(128, activation='tanh'),
#     Dense(64, activation='tanh'),
#     Dense(64, activation='tanh'),
#     Dense(32, activation='tanh'),
#     Dense(1, activation='tanh')
# ])

# optimizer = Adam(learning_rate=0.0001)
# model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# history = model.fit(
#     X_features, Y_norm,
#     epochs=100,
#     batch_size=32,
#     validation_split=0.2,
#     verbose=1
# )

# X_test = np.arange(-20, 20.1, 0.1)
# valid_test_mask = np.abs(X_test) <= radius
# X_test = X_test[valid_test_mask]
# X_test_norm = X_test / radius


# X_test = np.concatenate([X_test, X_test])
# X_test_norm = np.concatenate([X_test_norm, X_test_norm])
# signs_test = np.concatenate([np.ones_like(X_test_norm[:len(X_test_norm)//2]), 
#                            -np.ones_like(X_test_norm[:len(X_test_norm)//2])])


# X_test_features = np.column_stack([X_test_norm, signs_test])

# Y_pred = model.predict(X_test_features) * radius
# Y_pred = Y_pred.flatten()

# plt.figure(figsize=(10, 10))
# theta = np.linspace(0, 2*np.pi, 1000)
# circle_x = radius * np.cos(theta)
# circle_y = radius * np.sin(theta)
# plt.plot(circle_x, circle_y, 'g-', label='Круг', linewidth=2)

# plt.scatter(X_test, Y_pred, c='red', label='Предсказанные значения', alpha=0.7, s=20)

# plt.grid(True)
# plt.legend()
# plt.axis('equal')
# plt.title('Аппроксимация окружности')
# plt.show()

# Y_true_up = np.sqrt(radius**2 - X_test[:len(X_test)//2]**2)
# Y_true_down = -np.sqrt(radius**2 - X_test[:len(X_test)//2]**2)
# Y_true = np.concatenate([Y_true_up, Y_true_down])
# errors = np.abs(Y_pred - Y_true)

# plt.figure(figsize=(10, 5))
# plt.plot(X_test, errors)
# plt.title('Absolute Error vs X coordinate')
# plt.xlabel('X')
# plt.ylabel('Error')
# plt.grid(True)
# plt.show()

# Задание 4 - Синусоида

X = np.arange(-20, 20.1, 0.1)
X_features = np.column_stack([
    X,
    np.sin(X),
    np.cos(X),
    X % (2 * np.pi)
])

y = np.sin(X)

X_norm = X_features / np.maximum(1e-12, np.max(np.abs(X_features), axis=0))

split_idx = int(len(X_norm) * 0.8)
X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

model = Sequential([
    Dense(256, activation='relu', input_shape=(4,)),
    Dense(128, activation='tanh'),
    Dense(128, activation='relu'),
    Dense(64, activation='tanh'),
    Dense(64, activation='relu'),
    Dense(32, activation='tanh'),
    Dense(32, activation='relu'),
    Dense(16, activation='tanh'),
    Dense(16, activation='relu'),
    Dense(8, activation='tanh'),
    Dense(1)
])

optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

test_loss = model.evaluate(X_norm, y, verbose=0)
print(f'\nТочность на тестовой выборке (MSE): {test_loss[0]:.12f}')
print(f'MAE на тестовой выборке: {test_loss[1]:.12f}')

y_pred = model.predict(X_norm)

plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Реальные значения', alpha=0.5)
plt.scatter(X, y_pred, color='red', label='Предсказания', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Аппроксимация функции синуса')
plt.legend()
plt.grid(True)
plt.show()

# График ошибки
# plt.figure(figsize=(12, 6))
# plt.plot(history.history['loss'], label='Ошибка обучения')
# plt.plot(history.history['val_loss'], label='Ошибка валидации')
# plt.xlabel('Эпоха')
# plt.ylabel('MSE')
# plt.title('График ошибки обучения')
# plt.legend()
# plt.grid(True)
# plt.show()

max_error = np.max(np.abs(y_pred - y.reshape(-1, 1)))
print(f"\nМаксимальная абсолютная ошибка: {max_error:.12f}")