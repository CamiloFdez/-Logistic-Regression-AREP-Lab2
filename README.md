# Logistic Regression Arep Laboratory 2

## Table of Contents
- [1. Project Overview](#1-introduction)
- [2. Dataset Description](#2-dataset-description)
- [3. Exploratory Data Analysis](#3-exploratory-data-analysis)

---

## 1. Project Overview

This project involves creating a Logistic Regression model from scratch to make predictions about the risk of heart disease based on patient data.

This project involves the entire machine learning process, which includes the following steps:

- Exploratory Data Analysis
- Data Preprocessing
- Training the Model
- Visualization
- Regularization
- Deployment

This project aims to show the potential use of predictive models in the healthcare system to detect heart diseases early.

---

## 2. Dataset Description
The data set used is the Heart Disease Dataset, which can be found on Kaggle.

**Source:**
https://www.kaggle.com/datasets/neurocipher/heartdisease

**Dataset characteristics:**

* Total number of data points: 270
* Total number of features: 14
* The target variable is:
  * 1 if the patient has heart disease
  * 0 if the patient does not have heart disease


* Examples of the features used in the project:
  * Age: 29-77 years
  * Cholesterol: 112-564 mg/dl
  * Resting Blood Pressure
  * Maximum Heart Rate Achieved
  * ST Depression
  * Number of Major Vessels Colored by Fluoroscopy
* Data types: Numerical and Categorical
* Missing values: None

---

## 3. Exploratory Data Analysis
The initial exploration involved:

- Summary statistics of all features
- Checking for missing values
- Checking the distribution of classes
- Feature selection based on relevance to the medical field
- The numerical features were normalized to facilitate efficient gradient descent training.

The dataset was split into:

- 70% Training set
- 30% Test set (stratified to maintain class distribution)

---

