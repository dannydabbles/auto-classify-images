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


class AutoKerasClassify:
    """Class to manage the Auto Keras training process"""

    # TODO: Make options for these
    # Scale all images to squares with this dimension
    image_size = None
    # Number of images to process on each epoch
    batch_size = None
    # Set the time
    time = None
    # Infer the project directory from the location of this file
    project_dir = None
    # Set our image directory
    image_dir = None
    # Set our tensorboard logging directory
    tensorboard_dir = None
    # Set our auto_model directory for storing candidate autokeras model information
    auto_model_dir = None
    # Define our tensorboard callback
    tensorboard_callback = None

    def __init__(self,
                 image_size=75,
                 batch_size=32,
                 time=None,
                 project_dir=None,
                 image_dir=None,
                 tensorboard_dir=None,
                 auto_model_dir=None,
                 tensorboard_callback=None):
        self.image_size = image_size
        self.batch_size = batch_size
        if time is None:
            self.time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
        if project_dir is None:
            self.project_dir = os.path.dirname(os.path.realpath(__file__))
        if image_dir is None:
            self.image_dir = os.path.join(self.project_dir, "images")
        if tensorboard_dir is None:
            self.tensorboard_dir = os.path.join(self.project_dir, "logs/fit", self.time)
        if auto_model_dir is None:
            self.auto_model_dir = os.path.join(self.project_dir, "auto_model")
        if tensorboard_callback is None:
            self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.tensorboard_dir, histogram_freq=1)

    def generate_data(self):
        """Function to initialize our data generators"""

        datagen = ImageDataGenerator(
            # Reserve 30% of our data for validation
            validation_split=0.2,
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
            os.path.join(self.image_dir, "train"),
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
            target_size=(self.image_size, self.image_size))
        validate = datagen.flow_from_directory(
            # Select our image data directory
            os.path.join(self.image_dir, "train"),
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
            target_size=(self.image_size, self.image_size))
        datagen = ImageDataGenerator(
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
        testing = datagen.flow_from_directory(
            # Select our image data directory
            os.path.join(self.image_dir, "test"),
            # Always shuffle our data
            shuffle=True,
            # We're classifying things into categories (categorical cross-entropy style)
            class_mode='categorical',
            # Set our batch size of images for training the model (different from inception)
            # Note: This is set so high because we can't use a generator to generate images in several small batches
            #       So, we just get all our images at once.
            batch_size=50000,
            # Resize our images
            target_size=(self.image_size, self.image_size))

        # Calculate class weights for appropriate regularization
        counter = Counter(training.classes)
        max_val = float(max(counter.values()))
        class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}

        return training, validate, testing, class_weights

    def unpack_data(self, training, validate, testing, class_weights=None):
        """Since autokeras' fit() function does not accept data generators, let's unpack them"""

        # Unpack our validation data generator
        x_validate, y_validate = validate.next()
        x_validate = x_validate.astype('uint8')
        y_validate = y_validate.astype('uint8')

        # Unpack our test and validation data
        x_test, y_test = testing.next()
        x_test = x_test.astype('uint8')
        y_test = y_test.astype('uint8')

        # Print our labels
        label_map = dict((v, k) for k, v in training.class_indices.items())
        print("Labels: {}".format(label_map))
        # Write our labels to a file
        model_name = "autokeras_model_" + self.time
        model_path = os.path.join(self.project_dir, "models", model_name)
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

        return x_train, y_train, x_validate, y_validate, x_test, y_test, class_weights

    def generate_model(self, x_train, y_train, x_validate, y_validate, x_test, y_test, class_weights=None):
        """Use AutoKeras to run a Neural Architecture Search"""

        model = ak.ImageClassifier(
            # Select a random seed value
            seed=29,
            # Decide how many models to try out
            max_trials=25,
            # Set our auto_model directory for storing candidate models
            directory=self.auto_model_dir)
        model.fit(x_train, y_train,
                  # Perform regularization
                  class_weight=class_weights,
                  # Pass in our validation data
                  validation_data=(x_validate, y_validate),
                  # Set our callbacks
                  callbacks=[self.tensorboard_callback],
                  # Set our batch size per epoch
                  batch_size=self.batch_size)

        # The best model found during the Neural Architecture Search
        model_name = "autokeras_model_" + self.time
        model_path = os.path.join(self.project_dir, "models", model_name)
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
        print("INFO: Predictions is {}".format(model.predict(x_test)))

    def run(self):
        """Kick off training"""

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
        print("INFO: Training started with image size {} and batch size {}".format(self.image_size, self.batch_size))
        self.generate_model(*self.unpack_data(*self.generate_data()))
        end = datetime.now()
        delta = end - start
        print("INFO: Training completed in {}".format(str(delta)))


if __name__ == "__main__":
    AutoKerasClassify().run()
