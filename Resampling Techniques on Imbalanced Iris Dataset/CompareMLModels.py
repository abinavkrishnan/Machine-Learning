# EECS 658 Assignment 5
# Name: Abinav Krishnan
# Student ID: 3068700
# Date: 09/31/2024
# Description: Evaluate resampling techniques on an imbalanced Iris dataset using a neural network model.
# Inputs: None | Outputs: Confusion Matrices, Accuracy Scores (Standard, Class Balanced, and Balanced Accuracy)

# Imports
import pandas as pd  # For data handling
import numpy as np  # For numerical operations
from sklearn.model_selection import cross_val_predict  # For cross-validation
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score  # For evaluation metrics
from sklearn.neural_network import MLPClassifier  # Neural network model
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN  # Oversampling methods
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, TomekLinks  # Undersampling methods

# Load the dataset from a CSV file located in the same directory
# This dataset contains measurements (sepal length, sepal width, etc.) and class labels for flowers
dataset = pd.read_csv(
    "imbalanced_iris.csv", 
    names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
)

# Split the dataset into features (X) and labels (y)
X, Y = dataset.iloc[:, :4].values, dataset.iloc[:, 4].values

# Define the neural network model
# This model consists of two hidden layers with 10 neurons each and can train for up to 3000 iterations
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=3000, random_state=42)

# Function to train and evaluate the model using 2-fold cross-validation
# It prints the confusion matrix and accuracy score for each sampling method
def evaluate_model(X, y, description):
    y_pred = cross_val_predict(mlp, X, y, cv=2)  # Predict using 2-fold cross-validation
    print(f"\n{description} Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print(f"Accuracy Score: {accuracy_score(y, y_pred)}")

# Part 1: Evaluate the model on the imbalanced dataset without resampling
print("Part 1: Imbalanced Dataset")
y_pred = cross_val_predict(mlp, X, Y, cv=2)  # Perform 2-fold cross-validation

# Compute metrics for the imbalanced dataset
conf_matrix = confusion_matrix(Y, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)
print("Accuracy Score:", accuracy_score(Y, y_pred))
print("Class Balanced Accuracy:", np.mean(conf_matrix.diagonal() / conf_matrix.sum(axis=1)))
print("Balanced Accuracy (sklearn):", balanced_accuracy_score(Y, y_pred))

# Part 2: Apply oversampling techniques to balance the dataset
print("\nPart 2: Oversampling")
oversampling_methods = [
    (RandomOverSampler(), "Random Oversampling"),
    (SMOTE(), "SMOTE Oversampling"),
    (ADASYN(sampling_strategy='minority'), "ADASYN Oversampling")
]

# Evaluate the model using each oversampling method
for sampler, name in oversampling_methods:
    X_res, y_res = sampler.fit_resample(X, Y)  # Generate balanced dataset
    evaluate_model(X_res, y_res, name)  # Evaluate and print metrics

# Part 3: Apply undersampling techniques to reduce the majority class
print("\nPart 3: Undersampling")
undersampling_methods = [
    (RandomUnderSampler(), "Random Undersampling"),
    (ClusterCentroids(), "Cluster Centroids Undersampling"),
    (TomekLinks(), "Tomek Links Undersampling")
]

# Evaluate the model using each undersampling method
for sampler, name in undersampling_methods:
    X_res, y_res = sampler.fit_resample(X, Y)  # Generate reduced dataset
    evaluate_model(X_res, y_res, name)  # Evaluate and print metrics