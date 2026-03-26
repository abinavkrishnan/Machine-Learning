# EECS 658 Assignment 6
# Name: Abinav Krishnan
# Student ID: 3068700
# Date: 11/14/2024
# Description: The assignment involves applying k-means, GMM, and SOM clustering to the Iris dataset,
# analyzing performance with metrics and visualizations, and interpreting results.
# Inputs: None | Outputs: Confusion Matrices, Graphs

# Import needed libraries
import numpy as np  # Library for numerical operations
import pandas as pd  # Library for data manipulation and analysis
import matplotlib.pyplot as plt  # Library for plotting and data visualization
from sklearn.cluster import KMeans  # Import KMeans for k-means clustering
from sklearn.metrics import confusion_matrix, accuracy_score  # Import metrics for model evaluation
from scipy.optimize import linear_sum_assignment  # Import function for optimal assignment problem
from sklearn.preprocessing import LabelEncoder  # Import for encoding target labels
from sklearn.mixture import GaussianMixture  # Import GMM for clustering
from minisom import MiniSom  # Import MiniSom for SOM

# Load the Iris dataset
url = "iris.csv"  # File path for the Iris dataset
columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'species']  # Column names for dataset
data = pd.read_csv(url, names=columns)  # Load the CSV data into a DataFrame

# Separate the features (X) and target (y)
X = data[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']].values  # Extract feature columns
y = data['species'].values  # Extract target column

# Encode the target labels as numeric values
label_encoder = LabelEncoder()  # Initialize the label encoder
y_encoded = label_encoder.fit_transform(y)  # Encode the target labels

# --- Part 1: k-Means Clustering ---
print("Part 1: k-Means Clustering")

# List to store reconstruction errors for different k values
reconstruction_errors = []
k_values = range(1, 21)  # Test k values from 1 to 20

# Compute k-means for each k and record reconstruction error
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Initialize k-means with k clusters
    kmeans.fit(X)  # Fit to data
    reconstruction_errors.append(kmeans.inertia_)  # Record inertia as reconstruction error

# Plot reconstruction error vs. number of clusters
plt.figure()
plt.plot(k_values, reconstruction_errors, marker='o')
plt.title('Reconstruction Error vs. Number of Clusters (k)')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Reconstruction Error')
plt.grid(True)
plt.show()

# Identify the elbow point manually (e.g., k=3 for demonstration)
elbow_k = 3
print(f"Elbow found at k={elbow_k}")

# Fit k-means with the elbow point and predict cluster labels
kmeans_elbow = KMeans(n_clusters=elbow_k, random_state=42)
kmeans_elbow.fit(X)
predicted_labels_elbow = kmeans_elbow.predict(X)

# Compute confusion matrix for the elbow_k
conf_matrix_elbow = confusion_matrix(y_encoded, predicted_labels_elbow)
print("Confusion Matrix for k=elbow_k:")
print(conf_matrix_elbow)

# Calculate accuracy for elbow_k if it matches the number of true classes
if elbow_k == 3:
    cost_matrix = -conf_matrix_elbow  # Convert to cost matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Solve assignment problem
    reordered_conf_matrix = conf_matrix_elbow[:, col_ind]  # Reorder columns
    accuracy_elbow = np.trace(reordered_conf_matrix) / np.sum(reordered_conf_matrix)  # Compute accuracy
    print("Accuracy for k=elbow_k:", accuracy_elbow)
else:
    print("Cannot calculate Accuracy Score as classes and clusters do not match.")

# Fit k-means with k=3 and predict cluster labels
kmeans_3 = KMeans(n_clusters=3, random_state=42)
kmeans_3.fit(X)
predicted_labels_3 = kmeans_3.predict(X)

# Compute confusion matrix for k=3
conf_matrix_3 = confusion_matrix(y_encoded, predicted_labels_3)
print("Confusion Matrix for k=3:")
print(conf_matrix_3)

# Calculate accuracy for k=3
cost_matrix_3 = -conf_matrix_3
row_ind_3, col_ind_3 = linear_sum_assignment(cost_matrix_3)
reordered_conf_matrix_3 = conf_matrix_3[:, col_ind_3]
accuracy_3 = np.trace(reordered_conf_matrix_3) / np.sum(reordered_conf_matrix_3)
print("Accuracy for k=3:", accuracy_3)

# --- Part 2: Gaussian Mixture Models (GMM) ---
print("\nPart 2: Gaussian Mixture Models (GMM)")

# Store AIC and BIC values for different k
aic_scores, bic_scores = [], []

# Compute GMM for each k and record AIC and BIC
for k in k_values:
    gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
    gmm.fit(X)
    aic_scores.append(gmm.aic(X))
    bic_scores.append(gmm.bic(X))

# Plot AIC and BIC scores
plt.figure()
plt.plot(k_values, aic_scores, marker='o', label='AIC')
plt.plot(k_values, bic_scores, marker='x', label='BIC')
plt.title('AIC and BIC vs. Number of Components')
plt.xlabel('Number of Components (k)')
plt.ylabel('Scores')
plt.legend()
plt.grid(True)
plt.show()

# Identify elbows manually (set to k=3 for demonstration)
aic_elbow_k, bic_elbow_k = 3, 3
print(f"AIC elbow at k={aic_elbow_k}")
print(f"BIC elbow at k={bic_elbow_k}")

# Fit GMM with AIC elbow and compute metrics
gmm_aic = GaussianMixture(n_components=aic_elbow_k, covariance_type='diag', random_state=42)
gmm_aic.fit(X)
predicted_labels_aic = gmm_aic.predict(X)
conf_matrix_aic = confusion_matrix(y_encoded, predicted_labels_aic)
print("Confusion Matrix for k=aic_elbow_k:")
print(conf_matrix_aic)

if aic_elbow_k == 3:
    cost_matrix_aic = -conf_matrix_aic
    row_ind_aic, col_ind_aic = linear_sum_assignment(cost_matrix_aic)
    reordered_conf_matrix_aic = conf_matrix_aic[:, col_ind_aic]
    accuracy_aic = np.trace(reordered_conf_matrix_aic) / np.sum(reordered_conf_matrix_aic)
    print("Accuracy for k=aic_elbow_k:", accuracy_aic)

# --- Part 3: Self Organizing Map (SOM) ---
print("\nPart 3: Self Organizing Map (SOM)")

# Normalize features using min-max scaling
X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# Define SOM grid sizes
grid_sizes = [(3, 3), (7, 7), (15, 15), (25, 25)]
quantization_errors = []

# Train SOM for each grid size
for grid_size in grid_sizes:
    som = MiniSom(grid_size[0], grid_size[1], X_normalized.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
    som.train_random(X_normalized, 1000)
    quantization_error = som.quantization_error(X_normalized)
    quantization_errors.append(quantization_error)
    print(f"Quantization error for grid size {grid_size}: {quantization_error:.4f}")

    # Plot U-Matrix
    plt.figure(figsize=(8, 8))
    plt.title(f"U-Matrix for SOM {grid_size[0]}x{grid_size[1]}")
    plt.imshow(som.distance_map().T, cmap='bone_r')
    plt.colorbar()
    plt.show()

# Plot quantization error for grid sizes
grid_labels = [f"{size[0]}x{size[1]}" for size in grid_sizes]
plt.figure()
plt.plot(grid_labels, quantization_errors, marker='o')
plt.title('Quantization Error vs. Grid Size')
plt.xlabel('Grid Size')
plt.ylabel('Quantization Error')
plt.grid(True)
plt.show()
