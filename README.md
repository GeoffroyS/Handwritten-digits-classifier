# Handwritten digits classification using a Convolutional Neural Network

## Data visualisation


A sample of all 10 digits:

![0 to 9](https://i.imgur.com/w7ZADxi.png "A sample of all the digits from 0 to 9")

A sample of 10 different 8s from the MNIST dataset:

![8s](https://i.imgur.com/aZKllwM.png "A sample of different 8s from the MNIST dataset")

The distribution of all 10 digits in the dataset:

![digits distribution](https://i.imgur.com/0dfifFY.png "Distribution of all 10 digits in the dataset")

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

We keep the "best weights" and "best biases" and run the `evaluate` method on the test dataset.
Note that the "best weights/biases" are currently simply the weights/biases that got the best evaluation score on the validation set (this is to be changed, obviously)

```Evaluation for the test dataset: 9317 / 10000```
