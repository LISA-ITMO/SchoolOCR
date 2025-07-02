# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

# Загрузка данных
train = pd.read_csv("../input/emnist-balanced-train.csv", delimiter=',')
test = pd.read_csv("../input/emnist-balanced-test.csv", delimiter=',')
mapp = pd.read_csv("../input/emnist-balanced-mapping.txt", delimiter=' ',
                   index_col=0, header=None, squeeze=True)

# Константы
HEIGHT = 28
WIDTH = 28

# Функция для поворота изображений EMNIST
def rotate(image):
    image = image.reshape([HEIGHT, WIDTH])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image

# Выбираем только цифры (0-9) и букву X (код 33 в EMNIST balanced)
def filter_digits_and_x(data, labels):
    # Коды цифр 0-9: 0-9
    # Код буквы X: 33 (проверьте по вашему mapp)
    x_code = 33
    mask = (labels <= 9) | (labels == x_code)
    return data[mask], labels[mask]

# Применяем фильтрацию
train_x = train.iloc[:,1:]
train_y = train.iloc[:,0]
train_x, train_y = filter_digits_and_x(train_x, train_y)

test_x = test.iloc[:,1:]
test_y = test.iloc[:,0]
test_x, test_y = filter_digits_and_x(test_x, test_y)

# Поворот изображений
train_x = np.asarray(train_x)
train_x = np.apply_along_axis(rotate, 1, train_x)
test_x = np.asarray(test_x)
test_x = np.apply_along_axis(rotate, 1, test_x)

# Нормализация
train_x = train_x.astype('float32') / 255
test_x = test_x.astype('float32') / 255

# Переназначаем метки для буквы X (делаем её классом 10)
train_y[train_y == 33] = 10
test_y[test_y == 33] = 10

# One-hot encoding (теперь 11 классов: 0-9 + X)
num_classes = 11
train_y = np_utils.to_categorical(train_y, num_classes)
test_y = np_utils.to_categorical(test_y, num_classes)

# Reshape для CNN
train_x = train_x.reshape(-1, HEIGHT, WIDTH, 1)
test_x = test_x.reshape(-1, HEIGHT, WIDTH, 1)

# Разделение на train/val
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.10, random_state=7)

# Построение модели
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(5,5), padding='same', activation='relu',
                 input_shape=(HEIGHT, WIDTH,1)))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(train_x)
train_gen = datagen.flow(train_x, train_y, batch_size=32)

history = model.fit_generator(
    train_gen,
    steps_per_epoch=len(train_x)/32,
    epochs=10,
    validation_data=(val_x, val_y)
)

# Оценка модели
score = model.evaluate(test_x, test_y, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Визуализация нескольких примеров X
x_indices = np.where(np.argmax(test_y, axis=1) == 10)[0]
plt.figure(figsize=(10,5))
for i, idx in enumerate(x_indices[:10]):
    plt.subplot(2,5,i+1)
    plt.imshow(test_x[idx].squeeze(), cmap='gray')
    plt.title(f"Class: X")
    plt.axis('off')
plt.show()