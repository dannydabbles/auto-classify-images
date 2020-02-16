#!/usr/bin/python

# Regular imports
import os
from datetime import datetime
from PIL import Image
from IPython.display import display
import numpy as np
from collections import Counter

# Tensorflow/Keras imports
import tensorflow as tf
from tensorflow import lite
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import SGD, RMSprop


class InceptionClassify:
    # TODO: Make options for these
    # Scale all images to squares with this dimension
    image_size = None
    # Number of images to process on each epoch
    batch_size = None

    # Set the time
    time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

    # Infer the project directory from the location of this file
    project_dir = os.path.dirname(os.path.realpath(__file__))
    # Set our tensorboard logging directory
    tensorboard_dir = os.path.join(project_dir, "logs/fit", time)
    # Set our image directory
    image_dir = os.path.join(project_dir, 'images')

    # Define our tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)

    def __init__(self,
                 image_size=299,
                 batch_size=32,
                 time=None,
                 project_dir=None,
                 tensorboard_dir=None,
                 image_dir=None,
                 tensorboard_callback=None):
        self.image_size = image_size
        self.batch_size = batch_size
        if time is None:
            self.time = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
        if project_dir is None:
            self.project_dir = os.path.dirname(os.path.realpath(__file__))
        if tensorboard_dir is None:
            self.tensorboard_dir = os.path.join(self.project_dir, "logs/fit", self.time)
        if image_dir is None:
            self.image_dir = os.path.join(self.project_dir, 'images')
        if tensorboard_callback is None:
            self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.tensorboard_dir, histogram_freq=1)

    def generate_data(self):
        """Decorator function to initialize our data generators"""
        datagen = ImageDataGenerator(
            # Make all RGB values between -1 and 1 for InceptionV3
            preprocessing_function=preprocess_input,
            # Reserve 30% of our data for validation
            validation_split=0.2,
            # Rotate our pictures a little
            rotation_range=5,
            # Stretch our pictures a little
            width_shift_range=0.02,  #
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
            # Set our batch size of images per epoch
            batch_size=self.batch_size,
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
            # Set our batch size of images per epoch
            batch_size=self.batch_size,
            # Make generator spit out our validation set of images
            subset="validation",
            # Resize our images
            target_size=(self.image_size, self.image_size))
        datagen = ImageDataGenerator(
            # Make all RGB values between -1 and 1 for InceptionV3
            preprocessing_function=preprocess_input,
            # Rotate our pictures a little
            rotation_range=5,
            # Stretch our pictures a little
            width_shift_range=0.02,  #
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
            # Set our batch size of images per epoch
            batch_size=self.batch_size,
            # Resize our images
            target_size=(self.image_size, self.image_size))

        # Calculate class weights for appropriate regularization
        counter = Counter(training.classes)
        max_val = float(max(counter.values()))
        class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}

        return training, validate, testing, class_weights

    def train_model(self, training, validate, testing, class_weights):
        """Train our InceptionV3 model using transfer learning from pretrained imagenet weights"""

        # Print our labels
        label_map = dict((v, k) for k, v in training.class_indices.items())
        print("Labels: {}".format(label_map))
        # Write our labels to a file
        model_name = "inception_model_" + self.time
        model_path = os.path.join(self.project_dir, "models", model_name)
        with open(model_path + ".txt", "w") as labels_file:
            for label in sorted(label_map.keys()):
                labels_file.write("{}\n".format(label))

        # Count how many classes we have
        num_classes = len(label_map.keys())
        print("Num classes: {}".format(num_classes))

        # Check our training data by grabbing a single batch
        x_train, y_train = training.next()

        # Check the type of our data
        print("Data type: {}".format(type(x_train[0][0][0][0])))

        # Print some label data
        for i in range(min(int(len(x_train)), 10)):
            print("Label: " + label_map[np.where(y_train[i] == 1)[0][0]])
            # Note: This will not display on a terminal, but it's still useful to make
            #       sure that the images can (in theory) at least load
            display(Image.fromarray((x_train[i] * 127.5 + 127.5).astype('uint8'), 'RGB'))

        # create the base pre-trained model
        base_model = InceptionV3(
            weights='imagenet',
            layers=tf.keras.layers,  # See https://github.com/keras-team/keras/pull/9965#issuecomment-549126009
            input_tensor=Input(shape=(self.image_size, self.image_size, 3)),
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
            # layer.trainable = True

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(
            # Use the RMSprop optimizer
            optimizer=RMSprop(learning_rate=0.000005, rho=0.9),
            # Use categorical_crossentropy to calculate loss
            loss='categorical_crossentropy',
            # Keep track of our model's accuracy
            metrics=['accuracy'])

        # train the model on the new data for a few epochs
        history = model.fit(
            # Pass in our training data
            training,
            # Perform regularization
            class_weight=class_weights,
            # Number of epochs to train
            epochs=20,
            # Pass in our validation data
            validation_data=validate,
            # Set our callbacks
            callbacks=[self.tensorboard_callback])

        # The returned "history" object holds a record
        # of the loss values and metric values during training
        print('\nhistory dict:', history.history)

        # Print a summary of our model
        print(model.summary())

        # Save our partially trained model
        model_name = "inception_model_" + self.time
        model_path = os.path.join(self.project_dir, "models", model_name)
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
            # steps_per_epoch=1,
            epochs=20,
            validation_data=validate,
            # validation_steps=1,
            callbacks=[self.tensorboard_callback])

        # The returned "history" object holds a record
        # of the loss values and metric values during training
        print('\nhistory dict:', history.history)

        # Print a summary of our model
        print(model.summary())

        # Save our fully trained model
        model_name = "inception_model_" + self.time
        model_path = os.path.join(self.project_dir, "models", model_name)
        model.save(model_path + ".h5")
        tflite_model = lite.TFLiteConverter.from_keras_model(model).convert()
        open(model_path + ".tflite", "wb").write(tflite_model)

        # Evaluate the model on the test data using `evaluate`
        print('\n# Evaluate on test data')
        results = model.evaluate(testing)
        print('test loss, test acc:', results)

        # Generate some predictions
        print('\n# Generate predictions')
        print("INFO: Predictions is {}".format(model.predict(testing)))
        print('predictions shape:', predictions.shape)

    def run(self):
        # Print some debug information about our GPU
        print(tf.config.experimental.list_physical_devices(device_type=None))
        # Try and limit memory growth on our GPU(s) (if we have any)
        for gpu in tf.config.experimental.list_physical_devices(device_type='GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)

        # Train our model
        start = datetime.now()
        print("INFO: Training started with image size {} and batch size {}".format(self.image_size, self.batch_size))
        self.train_model(*self.generate_data())
        end = datetime.now()
        delta = end - start
        print("INFO: Training completed in {}".format(str(delta)))


if __name__ == "__main__":
    InceptionClassify().run()
