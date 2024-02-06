# Happy_Sad_Sentiment_Classification
Train a CNN to classify images as "happy" or "sad". Dataset includes labeled images for model training and evaluation. Contributions welcome to enhance emotion recognition through AI.


# HappySad Image Classification with CNN

## Overview

This project aims to build a Convolutional Neural Network (CNN) for classifying images as "happy" or "sad". The model is trained on a dataset containing labeled images for both emotions, enabling it to learn and make predictions.

## Folder Structure

```plaintext
HappySad_Image_Classification/
│
├── data/
│   ├── train/
│   │   ├── happy/
│   │   │   ├── happy_image1.jpg
│   │   │   ├── happy_image2.jpg
│   │   │   └── ...
│   │
│   └── test/
│       ├── happy/
│       │   ├── happy_test_image1.jpg
│       │   ├── happy_test_image2.jpg
│       │   └── ...
│       │
│       └── sad/
│           ├── sad_test_image1.jpg
│           ├── sad_test_image2.jpg
│           └── ...
│
├── models/
│   └── happy_sad_cnn_model.h5
│
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluation_metrics.py
│   ├── inference.py
│   └── ...
│
├── requirements.txt
│
└── README.md
