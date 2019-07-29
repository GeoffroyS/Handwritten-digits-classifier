# Handwritten digits classification using a Neural Network

## Getting started
### What?
A Neural Network to classify digits from the MNIST dataset

To do:
* some more data exploration/plotting
* work on the optimisation of the learning rate
* improve data normalisation

### How?
1. Download ubyte files from [Yann LeCun's page](http://yann.lecun.com/exdb/mnist/)
2. `pip install -r requirements.txt`

## Data visualisation

A sample of all 10 digits:

![0 to 9](https://i.imgur.com/w7ZADxi.png "A sample of all the digits from 0 to 9")

A sample of 10 different 8s from the MNIST dataset:

![8s](https://i.imgur.com/aZKllwM.png "A sample of different 8s from the MNIST dataset")

The distribution of all 10 digits in the training dataset:

![digits distribution](https://i.imgur.com/0dfifFY.png "Distribution of all 10 digits in the dataset")


## Training the NN
Let's create a NN with 784 (28px * 28px) neurons in the input layer, 30 in one hidden layer and 10 (for the values 0 to 9) in the output
layer: `nn = neuralnet.NeuralNet([784, 30, 10])`
and then train it:
`nn._stochastic_gd(training_data_list, 30, 10, 3.0, validation_data=validation_data_list)`

Evaluation on the validation dataset for all 30 epochs:
```
epoch 0: 7313 / 10000
epoch 1: 8241 / 10000
epoch 2: 8375 / 10000
epoch 3: 8398 / 10000
epoch 4: 8420 / 10000
.....
.....
epoch 25: 9464 / 10000
epoch 26: 9500 / 10000
epoch 27: 9517 / 10000
epoch 28: 9470 / 10000
epoch 29: 9498 / 10000
```

## Evaluation on the test dataset
We keep the "best weights" and "best biases" and run the `evaluate` method on the test dataset:
`test_data_eval = nn.evaluate(test_data_list)`
Note that the "best weights/biases" are currently simply the weights/biases that got the best evaluation score on the validation set (this is to be changed)

```Evaluation for the test dataset: 9317 / 10000```
