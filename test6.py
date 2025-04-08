import numpy as np
import os, shutil
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Input
from keras.optimizers import RMSprop
from scipy.signal import convolve2d
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_dir = 'C:\\Users\\gagilev_ae\\Desktop\\ai_part2\\ai_labs\\sem2\\data6\\tr'
# validation_dir = 'C:\\Users\\gagilev_ae\\Desktop\\ai_part2\\ai_labs\\sem2\\data6\\val'
train_dir = 'C:\\Users\\Red\\Desktop\\Rows\\data\\tr'
validation_dir = 'C:\\Users\\Red\\Desktop\\Rows\\data\\val'
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')



inp = Input((150, 150, 3))
lay = Conv2D(32, (3, 3), activation='relu')(inp)
lay = Flatten()(lay)
lay = Dense(80, activation='relu')(lay)
lay = Dense(80, activation='relu')(lay)
lay = Dense(80, activation='relu')(lay)
lay = Dense(80, activation='relu')(lay)
lay = Dense(80, activation='relu')(lay)
lay = Dense(80, activation='relu')(lay)
out = Dense(1, activation='linear')(lay)
model = Model(inp, out)


# model = PyDataset([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dense(1, activation='sigmoid'),
# ])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['acc'])
history = model.fit(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator, validation_steps=50)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()