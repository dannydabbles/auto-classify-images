#!/usr/bin/python

import os
from tensorflow import lite
import tensorflow as tf
import autokeras as ak
from datetime import datetime
from PIL import Image
from IPython.display import display
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

image_size = 75

project_dir = os.path.dirname(os.path.realpath(__file__))
auto_model_dir = os.path.join(project_dir, "auto_model")
image_dir = os.path.join(project_dir, "images")
tensorboard_dir = os.path.join(project_dir, "logs/fit", datetime.now().strftime("%Y%m%d-%H%M%S"))

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)


def generate_data(func):
    def _generate_data_fn():
        datagen = ImageDataGenerator(
            validation_split=0.3,
            rotation_range=5,
            width_shift_range=0.02,
            height_shift_range=0.02,
            shear_range=0.02,
            zoom_range=0.02,
            horizontal_flip=True,
            fill_mode='nearest')
        training = datagen.flow_from_directory(
            os.path.join(image_dir, "train"),
            shuffle=True,
            class_mode='categorical',
            batch_size=50000,
            subset="training",
            target_size=(image_size, image_size))
        validate = datagen.flow_from_directory(
            os.path.join(image_dir, "train"),
            shuffle=True,
            class_mode='categorical',
            batch_size=20000,
            subset="validation",
            target_size=(image_size, image_size))
        return func(training, validate)

    return _generate_data_fn


def unpack_data(func):
    def _unpack_data_fn(training=None, validate=None):
        x_validate, y_validate = validate.next()

        x_validate = x_validate.astype('uint8')
        y_validate = y_validate.astype('uint8')

        split_num = int(len(x_validate) * .25)
        x_test = x_validate[:split_num]
        y_test = y_validate[:split_num],
        x_vaidate = x_validate[split_num:]
        y_vaidate = y_validate[split_num:]

        label_map = dict((v, k) for k, v in training.class_indices.items())
        print(label_map)
        x_train, y_train = training.next()
        x_train = x_train.astype('uint8')
        y_train = y_train.astype('uint8')

        print("Data type: {}".format(type(x_train[0][0][0][0])))

        for i in range(min(int(len(x_train)), 10)):
            print("Label: " + label_map[np.where(y_train[i] == 1)[0][0]])
            display(Image.fromarray(x_train[i], 'RGB'))

        return func(x_train, y_train, x_validate, y_validate, x_test, y_test)

    return _unpack_data_fn


@generate_data
@unpack_data
def generate_model(x_train, y_train, x_validate, y_validate, x_test, y_test):
    model = ak.ImageClassifier(
        seed=29,
        max_trials=25,
        directory=auto_model_dir)
    model.fit(x_train, y_train,
              validation_data=(x_validate, y_validate),
              callbacks=[tensorboard_callback],
              batch_size=32)
    model_name = "autokeras_model_" + datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
    model_path = os.path.join(project_dir, "models", model_name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model = model.export_model()
    model.save(model_path + ".h5")
    tflite_model = lite.TFLiteConverter.from_keras_model(model).convert()
    open(model_path + ".tflite", "wb").write(tflite_model)
    print("INFO: Predictions is {}".format(model.predict(x_test)))
    return model.evaluate(x_test, y_test)


def main():
    print("INFO: Training started with size {}".format(image_size))

    print(tf.config.experimental.list_physical_devices(device_type=None))
    for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # if os.path.exists(auto_model_dir):
    #    print("INFO: Removing saved model directory: {}".format(auto_model_dir))
    #    shutil.rmtree(auto_model_dir)

    start = datetime.now()
    print("INFO: Training results: {}".format(generate_model()))
    end = datetime.now()
    delta = end - start
    print("INFO: Training completed in {}".format(str(delta)))


if __name__ == "__main__":
    main()
