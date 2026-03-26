# EECS 658 Assignment 3
# Name: Abinav Krishnan
# Date: 09 / 28 / 2024
# Description: Compare different ML models on the iris dataset using 2-fold cross-validation
# Inputs: N/A | Outputs: String displaying the overall accuracy of the classifier and confusion matrix

# imports
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load the iris dataset
url = "iris.csv"  # Directory to csv file
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] # List for the names of titles
dataset = read_csv(url, names=names) # Load dataset while matching the names using pandas

# Splitting data into features and target labels
X = dataset.iloc[:, 0:4].values  # Features (sepal/petal lengths and widths)
y = dataset.iloc[:, 4].values    # Target (class labels)

# Encode class labels for regression models
encoder = LabelEncoder()  # Create an encoder to transform class names into numbers
y_encoded = encoder.fit_transform(y)  # Transform the flower class names into numbers (0, 1, 2)

# Split data into 2 folds for training and testing
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y_encoded, test_size=0.50, random_state=1) # split 50/50

# Define the models
models = [
    ('Linear Regression', make_pipeline(PolynomialFeatures(degree=1), LinearRegression())), # Linear regression model, but using degree 1 polynomial features
    ('2 Degree Polynomial Regression', make_pipeline(PolynomialFeatures(degree=2), LinearRegression())), # Polynomial regression of degree 2 (adds squares of features)
    ('3 Degree Polynomial Regression', make_pipeline(PolynomialFeatures(degree=3), LinearRegression())), # Polynomial regression of degree 3 (adds cubes and cross terms of features)
    ('Naive Bayes', GaussianNB()), # Naive Bayes classification model
    ('kNN', KNeighborsClassifier()), # k-Nearest Neighbors classification model
    ('LDA', LinearDiscriminantAnalysis()), # Linear Discriminant Analysis classification model
    ('QDA', QuadraticDiscriminantAnalysis()) # Quadratic Discriminant Analysis classification model
]

# Function to fit the model, test on folds, and print the results
def evaluate_model(name, model, X_Fold1, X_Fold2, y_Fold1, y_Fold2):
    # Train on Fold 1 and test on Fold 2
    model.fit(X_Fold1, y_Fold1)  # Train the model using the first fold
    pred1 = model.predict(X_Fold2)  # Test the model on the second fold

    # Train on Fold 2 and test on Fold 1
    model.fit(X_Fold2, y_Fold2)  # Train the model using the second fold
    pred2 = model.predict(X_Fold1)  # Test the model on the first fold

    # Concatenate predictions and actual values from both folds
    actual = np.concatenate([y_Fold2, y_Fold1])  # Combine actual labels from both folds
    predicted = np.concatenate([pred1, pred2])  # Combine predictions from both folds

    # Ensure regression predictions are rounded and within valid range
    if "Regression" in name:  # Only for regression models
        predicted = np.round(predicted)  # Round predictions to nearest whole number
        predicted = np.clip(predicted, 0, 2)  # Make sure predictions stay between 0 and 2 (valid class labels)

    # Print results for this model
    print(f"Results for {name}:")  # Print the name of the model
    print("Accuracy:", accuracy_score(actual, predicted))  # Print accuracy of the model
    print("Confusion Matrix:\n", confusion_matrix(actual, predicted))  # Print confusion matrix
    print("\n")  # Add a blank line after the results for readability

# ---------------------------------------------------------------------------------------------------------------

# Additional imports
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

# Add the new models to the existing list
models.extend([
    ('SVM (LinearSVC)', LinearSVC(max_iter=10000)),  # Linear Support Vector Machine
    ('Decision Tree', DecisionTreeClassifier()),     # Decision Tree classifier
    ('Random Forest', RandomForestClassifier()),     # Random Forest classifier
    ('Extra Trees', ExtraTreesClassifier()),          # Extra Trees classifier
    ('Neural Network', MLPClassifier(max_iter=1000, random_state=1)) # Neural Network model
])

# Evaluate all models
for name, model in models:
    evaluate_model(name, model, X_Fold1, X_Fold2, y_Fold1, y_Fold2)  # Run the evaluation function for each model