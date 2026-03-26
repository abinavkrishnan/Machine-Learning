# EECS 658 Assignment 1
# Name: Abinav Krishnan
# Date: 08 / 30 / 2024
# Description: Supervised Machine Learning for iris dataset

# NBClassifier
import numpy as np 
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB

url = "iris.csv" # Directory to csv file
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'] # List for the names of titles
dataset = read_csv(url, names=names) # Load dataset while matching the names using pandas

array = dataset.values # get the values for the dataset
X = array[:, 0:4]  # contains flower features (sepal/petal length and width)
y = array[:, 4]    # contains flower class labels

# Split Data into 2 Folds for Training and Testing
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y, test_size=0.50, random_state=1) 

model = GaussianNB() # Make a model
model.fit(X_Fold1, y_Fold1) # Train on Fold 1
pred1 = model.predict(X_Fold2) # Test on Fold 2

model.fit(X_Fold2, y_Fold2) # Train on Fold 2
pred2 = model.predict(X_Fold1) # Test on Fold 1

# Concatenate actual and predicted classes
actual = np.concatenate([y_Fold2, y_Fold1]) # concatenate actual class
predicted = np.concatenate([pred1, pred2]) # concatenate predicted class

# Output the results
print("The overall accuracy of the classifier is", accuracy_score(actual, predicted))  # accuracy
print("\nConfusion Matrix:\n", confusion_matrix(actual, predicted)) # confusion matrix
print("\nThe sum of the confusion matrix is", confusion_matrix(actual, predicted).sum())  # sum
print("\n", classification_report(actual, predicted))  # Precision, Recall, & F1 Score
