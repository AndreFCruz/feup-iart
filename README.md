# FEUP-IART
This repository hosts our project for the Artificial Intelligence (IART) course unit of the Masters in Informatics and Computer Engineering at FEUP.

## Introduction
The aim of this project was to develop a binary classifier for identifying pulsars from a set of candidates.

Main difficulties include:
* balancing of the dataset (91% of candidates belong to the negative class, _i.e._ not pulsars);
* size of dataset, which includes 17898 learning samples: sizeable but small for deep learning purposes;

## Dataset
HTRU2 is a data set which describes a sample of pulsar candidates collected during the High Time Resolution Universe Survey (South).

Available [here](http://archive.ics.uci.edu/ml/datasets/HTRU2).

## Tools
* Python3
* Keras
* Tensorflow (backend)
* Scikit-Learn
* TensorBoard

## Install Dependencies
```
pip3 install -r requirements.txt
```

## Contributors
* [André Cruz](https://github.com/AndreFCruz)
* [Edgar Carneiro](https://github.com/EdgarACarneiro)
* [João Carvalho](https://github.com/jflcarvalho)