#!/usr/bin/python

import os
import shutil
import time
import traceback
import autokeras as ak
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import random
from IPython.display import display
import gc
import numpy as np
from collections import Counter

import tensorflow as tf
from tensorflow import lite
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Input
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, RMSprop

project_dir = os.path.dirname(os.path.realpath(__file__))

tensorboard_dir=os.path.join(project_dir, "logs/fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)

print(tf.config.experimental.list_physical_devices(device_type=None))
for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

image_size = 299
batch_size = 32
image_dir = os.path.join(project_dir, 'images')

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
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
    batch_size=batch_size,
    subset="training",
    target_size=(image_size, image_size))
validate = datagen.flow_from_directory(
    os.path.join(image_dir, "train"),
    shuffle=True,
    class_mode='categorical',
    batch_size=batch_size,
    subset="validation",
    target_size=(image_size, image_size))

label_map = dict((v,k) for k,v in training.class_indices.items())
print(label_map)

num_classes = len(label_map.keys())
print("Num classes: {}".format(num_classes))

x_train, y_train = training.next()

counter = Counter(training.classes)
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

print("Data type: {}".format(type(x_train[0][0][0][0])))

for i in range(min(int(len(x_train)), 10)):
    print("Label: " + label_map[np.where(y_train[i] == 1)[0][0]])
    display(Image.fromarray((x_train[i]*127.5+127.5).astype('uint8'), 'RGB'))

start = datetime.now()
print("INFO: Training started with image size {} and batch size {}".format(image_size, batch_size))

# create the base pre-trained model
base_model = InceptionV3(
    weights='imagenet',
    layers=tf.keras.layers, # See https://github.com/keras-team/keras/pull/9965#issuecomment-549126009
    input_tensor=Input(shape=(image_size, image_size, 3)),
    include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(num_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False
    #layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(
    optimizer=RMSprop(learning_rate=0.000005, rho=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# train the model on the new data for a few epochs
history = model.fit(
    training,
    class_weight=class_weights,
    #steps_per_epoch=1,
    epochs=15,
    validation_data=validate,
    #validation_steps=1,
    callbacks=[tensorboard_callback])

# The returned "history" object holds a record
# of the loss values and metric values during training
print('\nhistory dict:', history.history)

print(model.summary())

model_name = "inception_model_" + datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
model_path = os.path.join(project_dir, "models", model_name)
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path + "_half.h5")
tflite_model = lite.TFLiteConverter.from_keras_model(model).convert()
open(model_path + "_half.tflite", "wb").write(tflite_model)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(
    optimizer=SGD(lr=0.00005, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history = model.fit(
    training,
    class_weight=class_weights,
    #steps_per_epoch=1,
    epochs=15,
    validation_data=validate,
    #validation_steps=1,
    callbacks=[tensorboard_callback])

# The returned "history" object holds a record
# of the loss values and metric values during training
print('\nhistory dict:', history.history)

print(model.summary())

model_name = "inception_model_" + datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
model_path = os.path.join(project_dir, "models", model_name)
model.save(model_path + ".h5")
tflite_model = lite.TFLiteConverter.from_keras_model(model).convert()
open(model_path + ".tflite", "wb").write(tflite_model)

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(validate)
print('test loss, test acc:', results)

print('\n# Generate predictions for 3 samples')
print("INFO: Predictions is {}".format(model.predict(validate)))
print('predictions shape:', predictions.shape)

end = datetime.now()
delta = end - start
print("INFO: Training completed in {}".format(str(delta)))
