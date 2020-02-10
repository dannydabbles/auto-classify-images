help:
	@cat Makefile

BACKEND=tensorflow
PYTHON_VERSION?=3.6
CUDA_VERSION?=10.1
CUDNN_VERSION?=7
TEST=tests/
SRC?=$(shell pwd)
DATA?="$(SRC)/data"

build:
	docker build -t keras --build-arg python_version=$(PYTHON_VERSION) --build-arg cuda_version=$(CUDA_VERSION) --build-arg cudnn_version=$(CUDNN_VERSION) docker

bash: build
	docker run --gpus all -it --env KERAS_BACKEND=$(BACKEND) keras bash

ipython: build
	docker run --gpus all -it --env KERAS_BACKEND=$(BACKEND) keras ipython

notebook: build
	docker run --gpus all -it --net=host --env KERAS_BACKEND=$(BACKEND) keras bash -c ". ~/.bashrc && jupyter notebook --port=8888 --ip=0.0.0.0"

test: build
	docker run --gpus all -it --env KERAS_BACKEND=$(BACKEND) keras py.test $(TEST)

inception:
	docker run --gpus all -it --net=host --env KERAS_BACKEND=$(BACKEND) -v $(SRC):/src/auto_classify_images keras python /src/auto_classify_images/train_inception_model.py

autokeras:
	docker run --gpus all -it --net=host --env KERAS_BACKEND=$(BACKEND) -v $(SRC):/src/auto_classify_images keras python /src/auto_classify_images/train_autokeras_model.py
