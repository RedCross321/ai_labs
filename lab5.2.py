import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


X = np.arange(-20, 20.1, 0.1)
y = np.sin(X) + np.sin(X * np.sqrt(2))

T = np.floor(X / (2 * np.pi))
pos = X % (2 * np.pi)
X_features = np.column_stack((T, pos))

X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.25, random_state=13)

model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)
])

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
history = model.fit(X_train, y_train,epochs=200,batch_size=10)
train_loss = model.evaluate(X_train, y_train)
test_loss = model.evaluate(X_test, y_test)

# print(f'\nТочность на обучающей выборке (MSE): {train_loss[0]:.12f}')
# print(f'MAE на обучающей выборке: {train_loss[1]:.12f}')
# print(f'Точность на тестовой выборке (MSE): {test_loss[0]:.12f}')
# print(f'MAE на тестовой выборке: {test_loss[1]:.12f}')

plt.figure(figsize=(12, 6))
X_sorted_idx = np.argsort(X.flatten())
y_pred = model.predict(X_features)
plt.scatter(X.flatten()[X_sorted_idx], y_pred[X_sorted_idx], c='red', alpha=0.5, label='Предсказания')
plt.scatter(X.flatten()[X_sorted_idx], y[X_sorted_idx], c='blue', alpha=0.5, label='Исходные данные')
plt.title('Предсказания модели')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()