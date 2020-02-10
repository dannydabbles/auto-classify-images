#!/usr/bin/python

import os
from PIL import Image
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

for i, y in enumerate(y_train):
    x = x_train[i]
    y = y[0]

    image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images/train", cifar10_classes[y], "{}.jpg".format(i))
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    im = Image.fromarray(x)
    im.save(image_path)
