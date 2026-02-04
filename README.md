# Logistic Regression Arep Laboratory 2

## Table of Contents
- [1. Project Overview](#1-introduction)
- [2. Dataset Description](#2-dataset-description)
- [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
- [4. Logistic Regression Implementation](#4-logistic-regression-implementation)
- [5. Decision Boundary Visualization](#5-decision-boundary-visualization)
- [6. Regularization](#6-regularization)
- [7. Deployment with Amazon SageMaker](#7-deployment-with-amazon-sagemaker)
- [8. Conclusion](#8-conclusion)

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

![image](https://github.com/CamiloFdez/-Logistic-Regression-AREP-Lab2/blob/main/images/hearthdisease.PNG)

* Examples of the features used in the project:
  * Age: 29-77 years
  * Cholesterol: 112-564 mg/dl
  * Resting Blood Pressure
  * Maximum Heart Rate Achieved
  * ST Depression
  * Number of Major Vessels Colored by Fluoroscopy
* Data types: Numerical and Categorical
* Missing values: None

![image](https://github.com/CamiloFdez/-Logistic-Regression-AREP-Lab2/blob/main/images/boxplot.PNG)

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

![image](https://github.com/CamiloFdez/-Logistic-Regression-AREP-Lab2/blob/main/images/trainedvstested.PNG)

---

## 4. Logistic Regression Implementation

The model was built from scratch, and the techniques used were the Sigmoid activation function, Binary Cross-Entropy loss, Gradient Descent, and tracking the cost. The techniques for evaluating the model were accuracy, precision, recall, and F1 score. The training curves were used to ensure the convergence of the cost function.

Here is the cost vs iterations graph for the training process:

![image](https://github.com/CamiloFdez/-Logistic-Regression-AREP-Lab2/blob/main/images/costanditerations.PNG)

---

## 5. Decision Boundary Visualization

In order to better understand the process by which the model is distinguishing the classes, decision boundaries were plotted for combinations of clinical features such as:

- Age vs. Cholesterol
- Resting Blood Pressure vs. Max Heart Rate
- ST Depression vs. Number of Vessels

These visualizations helped illustrate the process by which some combinations of features were more distinguishing than others:

![image](https://github.com/CamiloFdez/-Logistic-Regression-AREP-Lab2/blob/main/images/desicionboundary.PNG)

---

## 6. Regularization

L2 regularization was also introduced to prevent overfitting and improve the model’s ability to generalize.

To test the effect of regularization, different values of λ were implemented as follows:

```python
λ = [0, 0.001, 0.01, 0.1, 1]
```

The cost vs iterations graph for the regularized model compared to the unregularized model is shown below:

![image](https://github.com/CamiloFdez/-Logistic-Regression-AREP-Lab2/blob/main/images/costcomparation.PNG)

To check the effect of regularization, the following factors were considered:

- Model performance metrics
- Weight magnitude
- Change in decision boundaries

Now here is the decision boundary with regularization (λ=0.1):
![image](https://github.com/CamiloFdez/-Logistic-Regression-AREP-Lab2/blob/main/images/desicionwithregularitation.PNG)

---

## 7. Deployment with Amazon SageMaker

---

## 8. Conclusion

The implementation of the Logistic Regression model was successful, and the model was able to predict the risk of heart diseases. The model was able to perform well on the test set, and the metrics for the model’s accuracy, precision, recall, and F1 score showed how effective the model was.

The addition of L2 regularization was successful in preventing overfitting, as can be seen through the metrics and decision boundary of the regularized and unregularized models. The regularized model was able to show better capabilities for generalizing the problem, making it a better model for the problem.

---