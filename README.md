# MAML_
This repo contains code adopted from cbfin's official implementation accompaning the paper  [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (Finn et al., ICML 2017)](https://arxiv.org/abs/1703.03400). It includes code for running the few-shot supervised learning domain experiments, including sinusoid regression, Omniglot classification, and MiniImagenet classification. 

### Abstract 
The goal of meta-learning is to train a model on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training samples. 
The proposed model uses an explicit training of the model’s parameters (optimization based inference) thanks to the gradient descent procedure. 
The task parameters are updated  with one or more gradient descent steps such that the performances on it improve once the model has been trained on few examples of the task.
The model neither add new parameters to learn nor  impose any constraint on the chosen architecture, it can be used with fully connected, convolutional and recurrent neural networks.
The goal is to build a few-shot learner so a learning model that quickly adapts to a new tasks with few samples and training steps

### Dependencies
This code requires the following:
* python 2.\* or python 3.\*
* TensorFlow v1.0+

### Data

For the Omniglot and MiniImagenet data, see the usage instructions in `data/omniglot_resized/resize_images.py` and `data/miniImagenet/proc_images.py` respectively.

### Usage
To run the code, see the usage instructions at the top of `main.py`.

