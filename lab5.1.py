import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X = np.arange(-20, 20.1, 0.1)
y = np.sin(X) + np.sin(X * np.sqrt(2))

X = X.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)

model = Sequential([
    Dense(128, activation='relu', input_shape=(1,)),
    Dense(128, activation='tanh'),
    Dense(64, activation='relu'),
    Dense(64, activation='tanh'),
    Dense(32, activation='relu'),
    Dense(32, activation='tanh'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
model.fit(X_train, y_train, epochs=150, batch_size=10, validation_split=0.2)
_, accuracy = model.evaluate(X_train, y_train)
_, accuracy2 = model.evaluate(model.predict(X_test), y_test)

print(f'\nТочность на тестовой выборке (MSE): {accuracy:.12f}')
print(f'MAE на тестовой выборке: {accuracy2:.12f}')

y_pred = model.predict(X_train)

plt.figure(figsize=(12, 6))
plt.scatter(X_train, y_train, color='blue', label='Реальные значения', alpha=0.5)
plt.scatter(X_train, y_pred, color='red', label='Предсказания', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Аппроксимация функции синуса')
plt.legend()
plt.grid(True)
plt.show()
