import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Input, Concatenate
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau


X = np.arange(-20, 20.1, 0.1)
y = np.sin(X) + np.sin(X * np.sqrt(2))


def create_periodic_features(x):
    features = np.column_stack([
        np.sin(x),
        np.cos(x),
        np.sin(x * np.sqrt(2)),
        np.cos(x * np.sqrt(2)),
    ])
    return features

def create_periodic_network():
    model = Sequential([
        Dense(64, activation='relu', input_shape=(1,)),
        Dense(32, activation='relu'),
        Dense(4, activation='linear', name='periodic_features')
    ])
    return model

def create_prediction_network():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(4,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    return model

x_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_scaled = x_scaler.fit_transform(X.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

periodic_features_train = create_periodic_features(X_train)
periodic_features_test = create_periodic_features(X_test)


print("Training periodic feature extraction network...")
periodic_network = create_periodic_network()
periodic_network.compile(optimizer=Adam(learning_rate=0.001), 
                       loss='mse',
                       metrics=['mae'])

periodic_history = periodic_network.fit(
    X_train,
    periodic_features_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

train_periodic_features = periodic_network.predict(X_train)
test_periodic_features = periodic_network.predict(X_test)

print("\nTraining final prediction network...")
prediction_network = create_prediction_network()
prediction_network.compile(optimizer=Adam(learning_rate=0.001),
                         loss='mse',
                         metrics=['mae'])

prediction_history = prediction_network.fit(
    train_periodic_features,
    y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

train_loss = prediction_network.evaluate(train_periodic_features, y_train)
test_loss = prediction_network.evaluate(test_periodic_features, y_test)

# print(f'\nТочность на обучающей выборке (MSE): {train_loss[0]:.12f}')
# print(f'MAE на обучающей выборке: {train_loss[1]:.12f}')
# print(f'Точность на тестовой выборке (MSE): {test_loss[0]:.12f}')
# print(f'MAE на тестовой выборке: {test_loss[1]:.12f}')


plt.figure(figsize=(20, 5))

extracted_features = periodic_network.predict(X_scaled)
true_features = create_periodic_features(X_scaled)

X_features = periodic_network.predict(X_scaled)
y_pred = prediction_network.predict(X_features)

plt.scatter(X, y, c='blue', alpha=0.5, label='Исходные данные')
plt.scatter(X, y_pred, c='red', alpha=0.5, label='Предсказания')
plt.title('Финальные предсказания')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()