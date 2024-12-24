import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# https://www.kaggle.com/datasets/uciml/iris?resource=download
data = pd.read_csv('iris.csv')
X = data.iloc[:, :-1].values

label_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = data.iloc[:, -1].map(label_mapping).values


X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

np.random.seed(42)
indices = np.random.permutation(len(X))
split_idx = int(len(X) * 0.8)
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train = X_norm[train_indices]
X_test = X_norm[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)


history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

n_classes = 3
conf_matrix = np.zeros((n_classes, n_classes))
for true, pred in zip(y_test_classes, y_pred_classes):
    conf_matrix[true][pred] += 1

class_names = ['setosa', 'versicolor', 'virginica']

plt.figure(figsize=(8, 6))

plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Матрица ошибок')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')

for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(int(conf_matrix[i, j]), 'd'),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

plt.tight_layout()
plt.show()
