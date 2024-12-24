import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

radius = 20
X_edges = np.linspace(-radius, -radius, 200)
X_edges = np.append(X_edges, np.linspace(radius, radius, 200))
X_middle = np.linspace(-radius, radius, 800)
X = np.concatenate([X_edges, X_middle])
X = np.sort(X)
valid_mask = np.abs(X) <= radius
X = X[valid_mask]

Y_up = np.sqrt(radius**2 - X**2)
Y_down = -np.sqrt(radius**2 - X**2)

X = np.concatenate([X, X])
Y = np.concatenate([Y_up, Y_down])
signs = np.concatenate([np.ones_like(Y_up), -np.ones_like(Y_down)])

X_norm = X / radius
Y_norm = Y / radius

X_features = np.column_stack([X_norm, signs])

indices = np.random.permutation(len(X))
X_features = X_features[indices]
Y_norm = Y_norm[indices]

model = Sequential([
    Dense(256, activation='tanh', input_shape=(2,)),
    Dense(256, activation='tanh'),
    Dense(256, activation='tanh'),
    Dense(128, activation='tanh'),
    Dense(128, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(32, activation='tanh'),
    Dense(1, activation='tanh')
])

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

history = model.fit(
    X_features, Y_norm,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

X_test = np.arange(-20, 20.1, 0.1)
valid_test_mask = np.abs(X_test) <= radius
X_test = X_test[valid_test_mask]
X_test_norm = X_test / radius


X_test = np.concatenate([X_test, X_test])
X_test_norm = np.concatenate([X_test_norm, X_test_norm])
signs_test = np.concatenate([np.ones_like(X_test_norm[:len(X_test_norm)//2]), 
                           -np.ones_like(X_test_norm[:len(X_test_norm)//2])])


X_test_features = np.column_stack([X_test_norm, signs_test])

Y_pred = model.predict(X_test_features) * radius
Y_pred = Y_pred.flatten()

plt.figure(figsize=(10, 10))
theta = np.linspace(0, 2*np.pi, 1000)
circle_x = radius * np.cos(theta)
circle_y = radius * np.sin(theta)
plt.plot(circle_x, circle_y, 'g-', label='Круг', linewidth=2)

plt.scatter(X_test, Y_pred, c='red', label='Предсказанные значения', alpha=0.7, s=20)

plt.grid(True)
plt.legend()
plt.axis('equal')
plt.title('Аппроксимация окружности')
plt.show()

Y_true_up = np.sqrt(radius**2 - X_test[:len(X_test)//2]**2)
Y_true_down = -np.sqrt(radius**2 - X_test[:len(X_test)//2]**2)
Y_true = np.concatenate([Y_true_up, Y_true_down])
errors = np.abs(Y_pred - Y_true)

plt.figure(figsize=(10, 5))
plt.plot(X_test, errors)
plt.title('Absolute Error vs X coordinate')
plt.xlabel('X')
plt.ylabel('Error')
plt.grid(True)
plt.show()