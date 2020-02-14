# auto-classify-images

A simple framework to automate image classification.  Simply fill up the image directory with labeled subdirectories, and then fill each subdirectory with representative images.  Then run one of the following two commands to begin training.

## Loading cifar10 example data

<code>$ python download_cifar10_images.py</code>

## Prerequisites
The container for this project is based on the nvidia/cuda docker implementation.  Thus, prerequisites for running this container are equivalent to those at https://hub.docker.com/r/nvidia/cuda/ .

Note: The container should still run with only a CPU (i.e. no GPU), albeit slowly.

## Train using Inceptionv3 transfer learning
<code>$ ./inception.sh</code>

## Train using AutoKeras Neural Architecture Search
<code>$ ./autokeras.sh</code>
