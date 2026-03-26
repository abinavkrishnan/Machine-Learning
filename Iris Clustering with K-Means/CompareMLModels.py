import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.spatial.distance import cdist

# Load the iris dataset from CSV file
url = "iris.csv"  # Path to the CSV file
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']  # Column names
dataset = read_csv(url, names=names)  # Load dataset with column names

# Split data into features (X) and target labels (y_true)
X = dataset.iloc[:, 0:4].values  # Feature columns
y_true = dataset.iloc[:, 4].values  # Class labels

# Step 1: Run k-means for k values from 1 to 20 to calculate reconstruction error
reconstruction_errors = []  # List to store reconstruction errors
k_values = range(1, 21)  # Range of k values to test

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Initialize k-means with current k
    kmeans.fit(X)  # Fit k-means to the data
    # Calculate average minimum distance to cluster centers (reconstruction error)
    reconstruction_errors.append(sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Step 2: Plot reconstruction error vs. number of clusters (k) to find the "elbow"
plt.plot(k_values, reconstruction_errors, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Reconstruction Error')
plt.title('Reconstruction Error vs. Number of Clusters (k)')
plt.show()

# Step 3: Identify the "elbow" point and use it for clustering
elbow_k = 3  # Manual input for elbow_k

# Step 4: Evaluate k-means clustering for elbow_k and k=3
for k in [elbow_k, 3]:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Initialize k-means with chosen k
    y_pred = kmeans.fit_predict(X)  # Cluster assignment for each data point
    
    if k == 3:
        # Map clusters to true labels by matching majority class in each cluster
        mapping = [np.argmax(np.bincount(y_true[y_pred == i])) for i in range(k)]
        y_pred_mapped = np.array([mapping[label] for label in y_pred])
        
        # Print confusion matrix and accuracy for k=3
        accuracy = accuracy_score(y_true, y_pred_mapped)
        print(f"\nConfusion Matrix for k={k}:\n", confusion_matrix(y_true, y_pred_mapped))
        print(f"Accuracy for k={k}: {accuracy:.2f}")
    else:
        # If k != 3, print confusion matrix without accuracy
        print(f"\nConfusion Matrix for k={elbow_k}:\n", confusion_matrix(y_true, y_pred))
        print("Cannot calculate Accuracy Score because the number of classes is not the same as the number of clusters.")
