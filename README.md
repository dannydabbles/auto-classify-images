# auto-classify-images

A simple framework to automate image classification.  Simply create the ./image/train and ./image/test directories with labeled subdirectories, and then fill each subdirectory with representative images.  Then run one of the two shell scripts below to begin training (or both).

## Loading cifar10 example data

<code>$ mkdir -p images/train # Optional</code>

<code>$ mkdir -p images/test # Optional</code>

<code>$ python download_cifar10_images.py # Download cifar10 files</code>

## Prerequisites
The container for this project is based on the nvidia/cuda docker implementation.  Thus, prerequisites for running this container are equivalent to those at https://hub.docker.com/r/nvidia/cuda/ .

Note: The container should still run with only a CPU (i.e. no GPU), albeit slowly.

## Train using Inceptionv3 transfer learning
<code>$ ./inception.sh</code>

## Train using AutoKeras Neural Architecture Search
<code>$ ./autokeras.sh</code>
