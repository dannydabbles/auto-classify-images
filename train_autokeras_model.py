#!/usr/bin/python

# Regular imports
import os
from collections import Counter
from datetime import datetime
from PIL import Image
from IPython.display import display
import numpy as np

# Tensorflow/Keras/AutoKeras imports
from tensorflow import lite
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import autokeras as ak

# TODO: Make options for these
# Scale all images to squares with this dimension
image_size = 75
# Number of images to process on each epoch
batch_size = 32

# Set the time
time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

# Infer the project directory from the location of this file
project_dir = os.path.dirname(os.path.realpath(__file__))
# Set our image directory
image_dir = os.path.join(project_dir, "images")
# Set our tensorboard logging directory
tensorboard_dir = os.path.join(project_dir, "logs/fit", time)
# Set our auto_model directory for storing candidate autokeras model information
auto_model_dir = os.path.join(project_dir, "auto_model")

# Define our tensorboard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)


def generate_data(func):
    """Decorator function to initialize our data generators"""
    def _generate_data_fn():
        datagen = ImageDataGenerator(
            # Reserve 30% of our data for validation
            validation_split=0.3,
            # Rotate our pictures a little
            rotation_range=5,
            # Stretch our pictures a little
            width_shift_range=0.02,
            height_shift_range=0.02,
            shear_range=0.02,
            zoom_range=0.02,
            # Flip our pictures
            horizontal_flip=True,
            # Fill in any edge gaps
            fill_mode='nearest')
        training = datagen.flow_from_directory(
            # Select our image data directory
            image_dir,
            # Always shuffle our data
            shuffle=True,
            # We're classifying things into categories (categorical cross-entropy style)
            class_mode='categorical',
            # Set our batch size of images for training the model (different from inception)
            # Note: This is set so high because we can't use a generator to generate images in several small batches
            #       So, we just get all our images at once.
            batch_size=50000,
            # Make generator spit out our training set of images
            subset="training",
            # Resize our images
            target_size=(image_size, image_size))
        validate = datagen.flow_from_directory(
            # Select our image data directory
            image_dir,
            # Always shuffle our data
            shuffle=True,
            # We're classifying things into categories (categorical cross-entropy style)
            class_mode='categorical',
            # Set our batch size of images for validating the model (different from inception)
            # Note: This is set so high because we can't use a generator to generate images in several small batches
            #       So, we just get all our images at once.
            batch_size=20000,
            # Make generator spit out our validation set of images
            subset="validation",
            # Resize our images
            target_size=(image_size, image_size))

        # Calculate class weights for appropriate regularization
        counter = Counter(training.classes)
        max_val = float(max(counter.values()))
        class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}

        return func(training, validate, class_weights)
    return _generate_data_fn


def unpack_data(func):
    """Since autokeras' fit() function does not accept data generators, let's unpack them"""
    def _unpack_data_fn(training, validate, class_weights):
        # Unpack our validation data generator
        x_validate, y_validate = validate.next()
        x_validate = x_validate.astype('uint8')
        y_validate = y_validate.astype('uint8')

        # Unpack our test and validation data
        split_num = int(len(x_validate) * .25)
        x_test = x_validate[:split_num]
        y_test = y_validate[:split_num],
        x_vaidate = x_validate[split_num:]
        y_vaidate = y_validate[split_num:]

        # Print our labels
        label_map = dict((v, k) for k, v in training.class_indices.items())
        print("Labels: {}".format(label_map))
        # Write our labels to a file
        model_name = "autokeras_model_" + time
        model_path = os.path.join(project_dir, "models", model_name)
        with open(model_path + ".txt", "w") as labels_file:
            for label in sorted(label_map.keys()):
                labels_file.write("{}\n".format(label))

        # Unpack our training data
        x_train, y_train = training.next()
        x_train = x_train.astype('uint8')
        y_train = y_train.astype('uint8')

        # Check the type of our data
        print("Data type: {}".format(type(x_train[0][0][0][0])))

        # Print some label data
        for i in range(min(int(len(x_train)), 10)):
            print("Label: " + label_map[np.where(y_train[i] == 1)[0][0]])
            # Note: This will not display on a terminal, but it's still useful to make
            #       sure that the images can (in theory) at least load
            display(Image.fromarray(x_train[i], 'RGB'))

        return func(x_train, y_train, x_validate, y_validate, x_test, y_test, class_weights)

    return _unpack_data_fn


@generate_data
@unpack_data
def generate_model(x_train, y_train, x_validate, y_validate, x_test, y_test, class_weights):
    """Use AutoKeras to run a Neural Architecture Search"""
    model = ak.ImageClassifier(
        # Select a random seed value
        seed=29,
        # Decide how many models to try out
        max_trials=25,
        # Set our auto_model directory for storing candidate models
        directory=auto_model_dir)
    model.fit(x_train, y_train,
              # Perform regularization
              class_weight=class_weights,
              # Pass in our validation data
              validation_data=(x_validate, y_validate),
              # Set our callbacks
              callbacks=[tensorboard_callback],
              # Set our batch size per epoch
              batch_size=batch_size)

    # The best model found during the Neural Architecture Search
    model_name = "autokeras_model_" + time
    model_path = os.path.join(project_dir, "models", model_name)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model = model.export_model()
    model.save(model_path + ".h5")
    tflite_model = lite.TFLiteConverter.from_keras_model(model).convert()
    open(model_path + ".tflite", "wb").write(tflite_model)

    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    results = model.evaluate(x_test, y_test)
    print('test loss, test acc:', results)

    # Generate some predictions
    print('\n# Generate predictions')
    print("INFO: Predictions is {}".format(model.predict(x_test, y_test)))


def main():
    # Print some debug information about our GPU
    print(tf.config.experimental.list_physical_devices(device_type=None))
    # Try and limit memory growth on our GPU(s) (if we have any)
    for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # TODO: Make an option for this
    # if os.path.exists(auto_model_dir):
    #    print("INFO: Removing saved model directory: {}".format(auto_model_dir))
    #    shutil.rmtree(auto_model_dir)

    start = datetime.now()
    print("INFO: Training started with image size {} and batch size {}".format(image_size, batch_size))
    generate_model()
    end = datetime.now()
    delta = end - start
    print("INFO: Training completed in {}".format(str(delta)))


if __name__ == "__main__":
    main()
