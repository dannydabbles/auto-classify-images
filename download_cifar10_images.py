#!/usr/bin/python

# Regular imports
import os
from PIL import Image

# Keras imports
from tensorflow.keras.datasets import cifar10

# List the cifar10 classes in order
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

def main():
    # Get our cifar10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Combine training and test data so we can dump everything to a single folder
    training = (x_train, y_train, "train")
    validation = (x_test, y_test, "test")

    for x, y, dirname in [training, validation]:
        # Step through our images
        for i, item in enumerate(y):
            # Grab our image data
            image = x[i]
            # Grab our label
            label = cifar10_classes[item[0]]

            # Generate an image path
            image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      "images",
                                      dirname,
                                      label,
                                      "{}.jpg".format(i))
            os.makedirs(os.path.dirname(image_path), exist_ok=True)

            # Save the image data to a file
            im = Image.fromarray(image)
            im.save(image_path)

        print("Downladed {} images".format(len(y)))

if __name__ == "__main__":
    main()
