import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Dropout, Input, Concatenate
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2

X = np.arange(-20, 20.1, 0.02)
y = np.sin(X) + np.sin(X * np.sqrt(2))

T1 = 2 * np.pi
T2 = 2 * np.pi / np.sqrt(2)

def find_common_period(T1, T2, max_multiplier=100):
    for i in range(1, max_multiplier):
        for j in range(1, max_multiplier):
            if abs(i*T1 - j*T2) < 0.0001:
                return i*T1
    return T1

common_period = find_common_period(T1, T2)

period_number = np.floor(X / common_period)
position_in_period = (X / common_period - period_number) * 2 * np.pi

sin_position = np.sin(position_in_period)
cos_position = np.cos(position_in_period)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(np.column_stack([
    X,
    period_number,
    position_in_period,
    sin_position,
    cos_position
]))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def create_model():
    model = Sequential([
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4), input_shape=(5,)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

model = create_model()

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[reduce_lr, early_stopping],
    verbose=1
)

train_loss = model.evaluate(X_train, y_train)
test_loss = model.evaluate(X_test, y_test)

print(f'\nТочность на обучающей выборке (MSE): {train_loss[0]:.12f}')
print(f'MAE на обучающей выборке: {train_loss[1]:.12f}')
print(f'Точность на тестовой выборке (MSE): {test_loss[0]:.12f}')
print(f'MAE на тестовой выборке: {test_loss[1]:.12f}')

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Ошибка обучения')
plt.plot(history.history['val_loss'], label='Ошибка валидации')
plt.title('История обучения')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 2, 2)
X_all = scaler.transform(np.column_stack([
    X,
    period_number,
    position_in_period,
    sin_position,
    cos_position
]))
y_pred = model.predict(X_all)

plt.scatter(X, y, c='blue', alpha=0.5, label='Исходные данные')
plt.scatter(X, y_pred, c='red', alpha=0.5, label='Предсказания')
plt.title('Предсказания модели')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()