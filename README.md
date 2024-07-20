# Email-Spam-Filtering
Here's the structured content for your GitHub README.md:

---

# Email Spam Filtering Project

This project aims to classify emails as spam or ham (non-spam) using various machine learning algorithms. We evaluate the performance of each model and select the best one based on accuracy.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Models](#models)
- [Model Evaluation](#model-evaluation)
- [Best Model Selection](#best-model-selection)
- [Email Prediction Function](#email-prediction-function)
- [User Input Prediction](#user-input-prediction)
- [Repository](#repository)

## Project Overview

This project demonstrates how to implement and compare different machine learning models for email spam filtering. We use various algorithms to classify emails and evaluate their performance to determine the best model.

## Dataset

The dataset used is `spam.csv`, which contains labeled emails indicating whether they are spam or ham.

## Data Preprocessing

1. **Load Dataset:**
   - Load `spam.csv` and extract relevant columns (`v1` for labels and `v2` for text).
   - Rename columns to `label` and `text`.

2. **Label Mapping:**
   - Map the labels to binary values: `ham` to `0` and `spam` to `1`.

3. **Data Splitting:**
   - Split the dataset into training and testing sets using an 80-20 split.

## Feature Extraction

- Use the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer to convert email text into numerical features. This transforms the text data into a format suitable for machine learning models.

## Models

We evaluate the following machine learning models:

- **Naive Bayes**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**

## Model Evaluation

For each model, we calculate the following performance metrics:

- **Accuracy:** The proportion of correct predictions.
- **Precision:** The proportion of true positive predictions among all positive predictions.
- **Recall:** The proportion of true positive predictions among all actual positives.
- **F1 Score:** The harmonic mean of precision and recall.

The evaluation results are printed for each model, along with a detailed classification report.

## Best Model Selection

The model with the highest accuracy is selected as the best model. This model is used for predicting whether new emails are spam or ham.

## Email Prediction Function

A function `predict_email` is defined to predict the class of a new email. The email content is transformed using the TF-IDF vectorizer, and the best model is used to make the prediction. The function returns "Spam" if the prediction is 1, and "Ham" if the prediction is 0.

## User Input Prediction

The code prompts the user to input the content of an email. The `predict_email` function is called with this input, and the prediction (spam or ham) is displayed.


## Thank you!...
