# Lip Reading in Native Language using Machine Learning

This project focuses on lip reading for the Hindi language using machine learning techniques. The primary objective was to create a custom dataset, extract meaningful features from lip movements, and classify spoken sentences using various machine learning models.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Feature Extraction](#feature-extraction)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Conclusion and Future Work](#conclusion-and-future-work)

## Introduction

Lip reading involves the process of understanding speech by visually interpreting the movements of the lips, face, and tongue. This project aimed to develop a machine learning model for lip reading in Hindi, achieving a comparable accuracy to existing deep learning models.

## Dataset

The dataset was created with 20 individuals speaking 21 different sentences. For the initial phase, we used data from 4 individuals. Each video was processed to extract 68 facial landmarks, focusing on points 49 to 68 to identify lip movements.

## Feature Extraction

Three types of features were extracted from the lip region:

1. **Local Binary Patterns (LBP):** Ranging from 0 to 58.
2. **Optical Flow:** Ranging from 0 to 3.
3. **Lip Features:** Ranging from 1 to 10.

These features were compiled into a CSV file, with each row representing the features corresponding to a specific spoken sentence label.

## Model Training and Evaluation

Various classifier models were tested, including:

- XGBoost
- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)
- CatBoost
- Random Forest

After hyperparameter tuning, the Random Forest classifier achieved the best accuracy of approximately 34%.

## Results

The Random Forest model, with hyperparameter tuning, achieved an accuracy of 34%. This performance is on par with the only research paper available on Hindi lip reading using deep learning, which also reported an accuracy of 34%.

## Conclusion and Future Work

This project demonstrates that machine learning techniques can achieve comparable accuracy to deep learning models in lip reading tasks. Future improvements may include:

- Expanding the dataset with more speakers and sentences.
- Incorporating advanced feature extraction techniques.
- Exploring ensemble methods and hybrid models combining machine learning and deep learning approaches.

