import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

X = np.arange(-20, 20.1, 0.1)
y = np.sin(X) + np.sin(X * np.sqrt(2))

T = np.floor(X / (2 * np.pi))
pos = X % (2 * np.pi)
X_features = np.column_stack((T, pos))


X = X.reshape(-1, 1)
# X_train, X_test, y_train, y_test = train_test_split(X, T, test_size=0.25, random_state=13)

# model = Sequential([
#     Dense(4, activation='linear', input_shape=(1,)),
#     Dense(1)
# ])

# model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
# model.fit(X_train, y_train, epochs=150, batch_size=10, validation_split=0.2)
# y_pred = model.predict(X_train)
# plt.figure(figsize=(12, 6))
# plt.scatter(X_train, y_train, color='blue', label='Реальные значения', alpha=0.5)
# plt.scatter(X_train, np.round(y_pred), color='red', label='Предсказания', alpha=0.5)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Аппроксимация функции синуса')
# plt.legend()
# plt.grid(True)
# plt.show()

X1_train, X1_test, y1_train, y1_test = train_test_split(X, pos, test_size=0.25, random_state=13)

X1_train -= X1_train.mean(axis=0)
X1_train /= X1_train.std(axis=0)
y1_train -= y1_train.mean(axis=0)
y1_train /= y1_train.std(axis=0)

model1 = Sequential([
    Dense(80, activation='relu', input_shape=(1,)),
    Dense(80, activation='relu'),
    Dense(80, activation='relu'),
    Dense(80, activation='relu'),
    Dense(80, activation='relu'),
    Dense(80, activation='relu'),
    Dense(80, activation='relu'),
    Dense(1, activation='linear')
])
ear = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True, verbose=1)
# optimizer = Adam(learning_rate=0.00001)
model1.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
model1.fit(X1_train, y1_train, epochs=500, batch_size=5, validation_split=0.2, callbacks=[ear])
y1_pred = model1.predict(X1_train)
plt.figure(figsize=(12, 6))
plt.scatter(X1_train, y1_train, color='blue', label='Реальные значения', alpha=0.5)
plt.scatter(X1_train, y1_pred, color='red', label='Предсказания', alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Аппроксимация функции синуса')
plt.legend()
plt.grid(True)
plt.show()

# X_features = np.column_stack((y_pred, y1_pred))
# X2_train, X2_test, y2_train, y2_test = train_test_split(X_features, y, test_size=0.25, random_state=13)

# model2 = Sequential([
#     Dense(64, activation='relu', input_shape=(2,)),
#     Dense(32, activation='relu'),
#     Dense(16, activation='relu'),
#     Dense(8, activation='relu'),
#     Dense(1)
# ])

# model2.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])
# history = model2.fit(X2_train, y2_train,epochs=200,batch_size=10)
# train_loss = model2.evaluate(X2_train, y2_train)
# test_loss = model2.evaluate(X2_test, y2_test)

# # print(f'\nТочность на обучающей выборке (MSE): {train_loss[0]:.12f}')
# # print(f'MAE на обучающей выборке: {train_loss[1]:.12f}')
# # print(f'Точность на тестовой выборке (MSE): {test_loss[0]:.12f}')
# # print(f'MAE на тестовой выборке: {test_loss[1]:.12f}')

# plt.figure(figsize=(12, 6))
# X_sorted_idx = np.argsort(X.flatten())
# y_pred = model2.predict(X_features)
# plt.scatter(X.flatten()[X_sorted_idx], y_pred[X_sorted_idx], c='red', alpha=0.5, label='Предсказания')
# plt.scatter(X.flatten()[X_sorted_idx], y[X_sorted_idx], c='blue', alpha=0.5, label='Исходные данные')
# plt.title('Предсказания модели')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# plt.tight_layout()
# plt.show()