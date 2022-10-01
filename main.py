import argparse
import time
from math import sqrt
from random import random
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from cv2 import cv2
from keras import layers

parser = argparse.ArgumentParser(description="Введите: <THICKNESS>, <DIVISIONS>, <EPOCHS>, <STEPS_PER_EPOCH>")
parser.add_argument("--THICKNESS", type=int, default=5, help="Толщина кривой безье")
parser.add_argument("--DIVISIONS", type=int, default=25, help="Кол-во разбивок кривой")
parser.add_argument("--LAYERS", type=int, default=6, help="Кол-во скрытых слоёв сети")
parser.add_argument("--FILTERS", type=int, default=256, help="Начальное кол-во фильтров в свёрточных слоях")
parser.add_argument("--KERNEL_SIZE", type=int, default=3, help="Размер ядра в свёрточных слоях")
parser.add_argument("--EPOCHS", type=int, default=5, help="Кол-во эпох обучения")
parser.add_argument("--STEPS_PER_EPOCH", type=int, default=100, help="Кол-во шагов обучения в течение одной эпохи")

args = parser.parse_args()

THICKNESS = int(args.THICKNESS)
DIVISIONS = int(args.DIVISIONS)
LAYERS = int(args.LAYERS)
FILTERS = int(args.FILTERS)
KERNEL_SIZE = int(args.KERNEL_SIZE)
EPOCHS = int(args.EPOCHS)
STEPS_PER_EPOCH = int(args.STEPS_PER_EPOCH)
NUM_OF_POINTS = 3  # Кол-во точек по которым строится кривая безье
IMAGE_SIZE = 128  # Размер изображения


def check_points(points: List[float]) -> bool:
    """ Проверяет, подходят ли заданные точки для генерации 'пригодной' кривой безье """
    x1, x2, x3, y1, y2, y3 = points[0], points[1], points[2], points[3], points[4], points[5]
    x1x3_middle = x1 - (x1 - x3) / 2
    y1y3_middle = y1 - (y1 - y3) / 2
    l1 = sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    l2 = sqrt((x2 - x1x3_middle) ** 2 + (y2 - y1y3_middle) ** 2)

    if 0.5 <= (l2 / l1) <= 1.0:
        return True
    return False


def generate_points(num_of_points: int) -> np.ndarray:
    """ Генерирует случайные точки и возвращает и с shape=(6, ) """

    while True:
        points = [random() for _ in range(num_of_points * 2)]
        if check_points(points):
            break

    return np.array(points, dtype=np.float32)


def tf_generate_points(shape):
    return tf.numpy_function(generate_points, [shape], [tf.float32])


def generate_xy(points: Tuple, t: float) -> List:
    """ Для заданного параметра t генерирует координаты [x, y] """
    dt = 1 - t
    x = dt ** 2 * points[0] + 2 * dt * t * points[1] + t ** 2 * points[2]
    y = dt ** 2 * points[3] + 2 * dt * t * points[4] + t ** 2 * points[5]

    return [x, y]


def generate_beziers_coords(points: Tuple) -> np.ndarray:
    """ Генерирует координаты для кривой безье и возвращает их с shape=(DIVISIONS, 2) """
    # Генерируем параметр t. Кол-во точек = DIVISIONS, чтобы график выглядел гладким
    t = np.linspace(0, 1, DIVISIONS)

    coords = np.array([generate_xy(points, t_) for t_ in t])
    coords *= IMAGE_SIZE
    coords = coords.astype(np.int16)

    return coords


def tf_generate_beziers_coords(points):
    return points, tf.numpy_function(generate_beziers_coords, [points], [tf.int16])


def generate_image(beziers_coords: np.ndarray) -> List:
    """ Генерирует кривую безье и возвращает тензор с shape=(IMAGE_SIZE*IMAGE_SIZE) """
    beziers_coords = beziers_coords.astype(int)  # Чтобы заработало с .polylines
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    new_img = cv2.polylines(img, [beziers_coords], isClosed=False, color=1, thickness=THICKNESS)
    new_img = new_img.reshape(1, IMAGE_SIZE, IMAGE_SIZE)
    return new_img


def tf_generate_image(points, beziers_coords):
    points = tf.reshape(points,
                        shape=(-1, 6))
    img = tf.reshape(tf.numpy_function(generate_image, [beziers_coords], tf.double),
                     shape=(IMAGE_SIZE, IMAGE_SIZE))
    return points, img


def get_dataset(num_of_points: int):
    dataset = tf.data.Dataset.from_tensors(num_of_points) \
        .map(tf_generate_points) \
        .map(tf_generate_beziers_coords) \
        .map(tf_generate_image) \
        .repeat() \
        .batch(32)

    return dataset


def get_model():
    input_ = layers.Input(shape=(6,))

    curr_image_size = 4
    net = layers.Dense(FILTERS * curr_image_size * curr_image_size)(input_)
    net = layers.Reshape((curr_image_size, curr_image_size, FILTERS))(net)

    for i in range(LAYERS):
        if FILTERS / 2 == 0:
            break
        net = layers.Conv2DTranspose(filters=FILTERS / 2 ** i, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), padding='same',
                                     activation='relu', use_bias=True)(net)
        if curr_image_size < IMAGE_SIZE:  # Увеличиваем изображение, но не выходим за рамки:)
            net = layers.UpSampling2D(size=(2, 2))(net)
            curr_image_size *= 2

    if curr_image_size != IMAGE_SIZE:
        print(f"Недостаточно скрытых слоёв. curr_image_size= {curr_image_size} != {IMAGE_SIZE}")
        exit(-1)

    output = layers.Conv2DTranspose(filters=1, kernel_size=(KERNEL_SIZE, KERNEL_SIZE), padding='same',
                                    activation='sigmoid', use_bias=True)(net)

    model = tf.keras.Model(inputs=input_, outputs=output)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def _main() -> None:
    dataset = get_dataset(NUM_OF_POINTS)
    model = get_model()

    model.fit(dataset,
              epochs=EPOCHS,
              steps_per_epoch=STEPS_PER_EPOCH)

    model.save(time.strftime("%d.%m_%H.%M.h5"))

    for test in dataset.as_numpy_iterator():
        x_test = test[0][0]
        y_test = test[1][0]
        output = model.predict(x_test)

        plt.cla()
        plt.subplot(1, 2, 1)
        plt.title("Generated algorithm")
        plt.imshow(y_test.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap=plt.cm.binary)

        plt.subplot(1, 2, 2)
        plt.title("Generated by a neural network")
        plt.imshow(output.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap=plt.cm.binary)

        plt.waitforbuttonpress()


if __name__ == '__main__':
    _main()
