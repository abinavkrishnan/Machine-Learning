# EECS 658 Assignment 4
# Name: Abinav Krishnan
# Date: 09/28/2024
# Description: Compares PCA, simulated annealing, and genetic algorithm on the Iris dataset using 2-fold cross-validated Decision Tree.
# Inputs: N/A | Outputs: Accuracy, confusion matrix, features used, eigenvalues, PoV, and iteration logs.

# imports
import numpy as np
from pandas import read_csv
import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the iris dataset
url = "iris.csv"  # Path to csv file
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Splitting data into features and target labels
X = dataset.iloc[:, 0:4].values  # Features
y = dataset.iloc[:, 4].values    # Class labels

# Encode class labels to integers for model training
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data into two folds
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y_encoded, test_size=0.50, random_state=1)

# Initialize the Decision Tree classifier
model = DecisionTreeClassifier()

# Function to train, predict, and evaluate using 2-fold cross-validation
def evaluate_decision_tree(model, X_Fold1, X_Fold2, y_Fold1, y_Fold2):
    print("===================================")
    print("PART 1: ORIGINAL FEATURES")
    print("===================================")
    
    # Train on Fold 1, test on Fold 2
    model.fit(X_Fold1, y_Fold1)
    pred1 = model.predict(X_Fold2)

    # Train on Fold 2, test on Fold 1
    model.fit(X_Fold2, y_Fold2)
    pred2 = model.predict(X_Fold1)

    # Concatenate predictions and actual values
    actual = np.concatenate([y_Fold2, y_Fold1])
    predicted = np.concatenate([pred1, pred2])

    # Calculate and display accuracy and confusion matrix
    print("Decision Tree Classifier Results:")
    print("Accuracy:", accuracy_score(actual, predicted))
    print("Confusion Matrix:\n", confusion_matrix(actual, predicted))
    print("Features used: ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']\n")

# Evaluate the Decision Tree model
evaluate_decision_tree(model, X_Fold1, X_Fold2, y_Fold1, y_Fold2)

# Part 2: PCA Transformation

# Encode class labels to integers for compatibility
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Create PCA instance and fit it to data
pca = PCA(n_components=4)
pca.fit(X)

# Get eigenvalues, eigenvectors, and calculate PoV
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_
sum_eigenvalues = np.sum(eigenvalues)
PoV = np.cumsum(eigenvalues) / sum_eigenvalues

# Determine how many eigenvectors are needed for PoV > 0.90
num_components_needed = np.argmax(PoV >= 0.90) + 1

# Display eigenvalues and PoV in a neat table format
print("===================================")
print("PART 2: PCA TRANSFORMATION")
print("===================================")

# Display eigenvectors
print("Eigenvectors:")
for i, eigenvector in enumerate(eigenvectors, start=1):
    print(f"Eigenvector {i}: {eigenvector}")
    
print(f"\n{'Lambda':<10} {'Eigenvalue':<15} {'PoV including this eigenvalue and the ones above it'}")
for i, (eigenvalue, pov) in enumerate(zip(eigenvalues, PoV), start=1):
    print(f"{i:<10} {eigenvalue:<15.8f} {pov * 100:.0f}%")
print(f"{'Sum =':<10} {sum_eigenvalues:<15.8f}\n")

# Display the required number of eigenvectors
print(f"Number of eigenvectors required to achieve PoV > 0.90: {num_components_needed}\n")

# Transform the data into principal components
principalComponents = pca.transform(X)

# Use the first 'num_components_needed' components for transformation
W = eigenvectors[:num_components_needed].T  # Transpose to align dimensions
Z = principalComponents[:, :num_components_needed]  # Use selected components

# Split transformed data into 2 folds
Z_Fold1, Z_Fold2, y_Fold1, y_Fold2 = train_test_split(Z, y_encoded, test_size=0.50, random_state=1)

# Initialize Decision Tree classifier for PCA-transformed data
model_pca = DecisionTreeClassifier()

# Evaluate the Decision Tree model on PCA-transformed data
def evaluate_pca_model(model, Z_Fold1, Z_Fold2, y_Fold1, y_Fold2):
    # Train on Fold 1, test on Fold 2
    model.fit(Z_Fold1, y_Fold1)
    pred1 = model.predict(Z_Fold2)

    # Train on Fold 2, test on Fold 1
    model.fit(Z_Fold2, y_Fold2)
    pred2 = model.predict(Z_Fold1)

    # Combine predictions and actual values from both folds
    actual = np.concatenate([y_Fold2, y_Fold1])
    predicted = np.concatenate([pred1, pred2])

    # Display results
    print("Decision Tree on PCA-Transformed Data Results:")
    print("Accuracy:", accuracy_score(actual, predicted))
    print("Confusion Matrix:\n", confusion_matrix(actual, predicted))
    
    features = []
    for i in range(num_components_needed):
        letter = "z" + str(i+1)
        features.append(letter)
    print("Features used:", features , "\n")

# Evaluate the Decision Tree model for PCA-transformed data
evaluate_pca_model(model_pca, Z_Fold1, Z_Fold2, y_Fold1, y_Fold2)

# Part 3

# Combine the original features with the PCA-transformed features (8 total features)
Z_all = np.hstack((X, principalComponents))

# Define feature names in the order they appear in Z_all
feature_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'z1', 'z2', 'z3', 'z4']

# Define the simulated annealing parameters
iterations = 100
restart_value = 10
c = 1  # c in Pr[accept] = 1

# Helper function to generate a random subset of 1-2 features to perturb
def perturb_features(current_features):
    new_features = current_features.copy()
    indices_to_perturb = random.sample(range(len(current_features)), random.randint(1, 2))
    for idx in indices_to_perturb:
        new_features[idx] = not new_features[idx]  # Toggle feature inclusion
    return new_features

# Evaluate the model with a given subset of features
def evaluate_subset(features_subset, Z_all, y):
    # Select only the active features
    selected_features = np.where(features_subset)[0]
    Z_selected = Z_all[:, selected_features]

    # Split the selected features into two folds
    Z_Fold1, Z_Fold2, y_Fold1, y_Fold2 = train_test_split(Z_selected, y, test_size=0.50, random_state=1)

    # Initialize and train the Decision Tree model
    model = DecisionTreeClassifier()
    model.fit(Z_Fold1, y_Fold1)
    pred1 = model.predict(Z_Fold2)

    # Train on the second fold and predict on the first fold
    model.fit(Z_Fold2, y_Fold2)
    pred2 = model.predict(Z_Fold1)

    # Combine predictions and actual values from both folds
    actual = np.concatenate([y_Fold2, y_Fold1])
    predicted = np.concatenate([pred1, pred2])

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(actual, predicted)
    cm = confusion_matrix(actual, predicted)

    return accuracy, cm, selected_features

# Simulated Annealing algorithm
def simulated_annealing(Z_all, y, iterations, restart_value, c):
    # Start with all features included
    current_features = [True] * Z_all.shape[1]
    current_accuracy, _, _ = evaluate_subset(current_features, Z_all, y)
    best_features = current_features.copy()
    best_accuracy = current_accuracy

    restart_counter = 0  # Counter for restart logic

    print("===================================")
    print("PART 3: SIMULATED ANNEALING")
    print("===================================")

    # Iterate for the specified number of iterations
    for iteration in range(iterations):
        # Perturb the current feature set randomly
        new_features = perturb_features(current_features)
        new_accuracy, _, _ = evaluate_subset(new_features, Z_all, y)

        # Calculate the acceptance probability
        pr_accept = min(1, np.exp(c * (new_accuracy - current_accuracy)))

        # Generate a random uniform value
        random_uniform = random.uniform(0, 1)

        # Determine the status of the iteration
        if new_accuracy > best_accuracy:
            status = "Improved"
            best_features = new_features
            best_accuracy = new_accuracy
            current_features = new_features
            current_accuracy = new_accuracy
        elif random_uniform < pr_accept:
            status = "Accepted"
            current_features = new_features
            current_accuracy = new_accuracy
        else:
            status = "Discarded"

        # Map indices to feature names
        selected_feature_names = [feature_names[i] for i in np.where(new_features)[0]]

        # Print the iteration details
        print(
            f"Iteration {iteration + 1:3} | "
            f"Features: {selected_feature_names} | "
            f"Accuracy: {new_accuracy:.4f} | "
            f"Pr[accept]: {pr_accept:.4f} | "
            f"Random: {random_uniform:.4f} | "
            f"Status: {status}"
        )

        # Restart logic
        if status == "Improved":
            restart_counter = 0  # Reset the counter on improvement
        else:
            restart_counter += 1

        # Restart if necessary
        if restart_counter >= restart_value:
            print("Restart triggered!\n")
            current_features = [True] * Z_all.shape[1]  # Reset to all features
            current_accuracy, _, _ = evaluate_subset(current_features, Z_all, y)
            restart_counter = 0

    # Final evaluation with the best feature subset
    final_accuracy, final_cm, final_selected_features = evaluate_subset(best_features, Z_all, y)

    # Map indices to feature names for final selected features
    final_feature_names = [feature_names[i] for i in final_selected_features]

    # Print final results
    print("\nBest feature subset:", final_feature_names)
    print(f"Best accuracy: {best_accuracy:.4f}\n")
    print("Final Confusion Matrix:\n", final_cm)
    print("Final Accuracy Metric:", final_accuracy)
    print("Features used:", final_feature_names, "\n")

# Run the simulated annealing algorithm for Part 3
simulated_annealing(Z_all, y_encoded, iterations, restart_value, c)

# Part 4
# Function to decode a chromosome to a feature subset
def decode_chromosome(chromosome):
    return [i for i, gene in enumerate(chromosome) if gene == 1]

# Function to evaluate a feature subset using 2-fold cross-validation
def evaluate_chromosome(chromosome, Z_all, y):
    selected_features = decode_chromosome(chromosome)

    # Check if no features are selected to avoid invalid input
    if len(selected_features) == 0:
        return 0, np.zeros((len(np.unique(y)), len(np.unique(y))))

    Z_selected = Z_all[:, selected_features]

    # Split data into two folds
    Z_Fold1, Z_Fold2, y_Fold1, y_Fold2 = train_test_split(Z_selected, y, test_size=0.50, random_state=1)

    # Train and evaluate the model
    model = DecisionTreeClassifier()
    model.fit(Z_Fold1, y_Fold1)
    pred1 = model.predict(Z_Fold2)

    model.fit(Z_Fold2, y_Fold2)
    pred2 = model.predict(Z_Fold1)

    actual = np.concatenate([y_Fold2, y_Fold1])
    predicted = np.concatenate([pred1, pred2])

    accuracy = accuracy_score(actual, predicted)
    return accuracy, confusion_matrix(actual, predicted)

# Mutation function to flip a bit in the chromosome
def mutate(chromosome, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]  # Flip the bit

# Crossover function to combine two parent chromosomes
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Genetic Algorithm Main Loop
def genetic_algorithm(Z_all, y, initial_population, generations=50):
    population = initial_population

    print("\n===================================")
    print("PART 4: GENETIC ALGORITHM")
    print("===================================")

    for generation in range(generations):
        # Evaluate the population and sort by accuracy
        evaluated_population = [
            (chrom, evaluate_chromosome(chrom, Z_all, y)[0])
            for chrom in population if sum(chrom) > 0  # Ensure valid chromosomes
        ]
        evaluated_population.sort(key=lambda x: x[1], reverse=True)

        print(f"Generation {generation + 1}")
        for i, (chrom, acc) in enumerate(evaluated_population[:5]):
            selected_features = decode_chromosome(chrom)
            # Map indices to feature names
            selected_feature_names = [feature_names[i] for i in selected_features]
            print(f"  Set {i + 1}: Features {selected_feature_names}, Accuracy: {acc:.4f}")

        # Generate the next population using crossover and mutation
        next_population = []
        while len(next_population) < len(population):
            parents = random.sample(evaluated_population[:3], 2)  # Select top 3 parents randomly
            child1, child2 = crossover(parents[0][0], parents[1][0])
            mutate(child1)
            mutate(child2)
            next_population.extend([child1, child2])

        # Keep the population size constant
        population = next_population[:len(initial_population)]

    # Final evaluation of the best chromosome
    best_chromosome = evaluated_population[0][0]
    best_accuracy, best_cm = evaluate_chromosome(best_chromosome, Z_all, y)

    # Map indices to feature names for the best chromosome
    best_feature_names = [feature_names[i] for i in decode_chromosome(best_chromosome)]

    print("\nBest Feature Subset:", best_feature_names)
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print("Confusion Matrix:\n", best_cm)

# Combine original and PCA-transformed features for Z_all
Z_all = np.hstack((X, principalComponents))

# Initial population (as per the assignment instructions)
initial_population = [
    [0, 0, 0, 0, 1, 0, 0, 0],  # z1
    [0, 0, 0, 0, 1, 1, 0, 0],  # z1, z2
    [0, 0, 0, 0, 1, 1, 1, 0],  # z1, z2, z3
    [0, 0, 0, 0, 1, 1, 1, 1],  # z1, z2, z3, z4
    [1, 0, 0, 0, 0, 0, 0, 0],  # sepal-length
]

# Run the genetic algorithm
genetic_algorithm(Z_all, y_encoded, initial_population, generations=50)
