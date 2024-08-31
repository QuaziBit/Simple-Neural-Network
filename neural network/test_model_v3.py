import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

def load_normalized_mnist(file_path):
    print(f"{datetime.now()} - Loading normalized MNIST dataset from {file_path}...")
    data = pd.read_csv(file_path, header=None)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values
    print(f"{datetime.now()} - Normalized MNIST dataset loaded.")
    return X, y

# Function to load the normalized images from CSV files
def load_data_from_csv(output_dir):
    X = []
    y = []
    for digit in range(10):
        digit_dir = os.path.join(output_dir, f"{digit}_normalized")
        
        for filename in os.listdir(digit_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(digit_dir, filename)
                image_data = np.loadtxt(file_path, delimiter=",").flatten()
                X.append(image_data)
                y.append(digit)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"{datetime.now()} - Loaded {len(X)} images from {output_dir}")
    return X, y

# Activation functions
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# Forward propagation
def forward_propagation(inputs, weights):
    activations = [inputs]
    for i, weight_matrix in enumerate(weights):
        if i == len(weights) - 1:
            # Use softmax for the output layer
            inputs = softmax(np.dot(inputs, weight_matrix))
        else:
            # Use leaky_relu for hidden layers
            inputs = leaky_relu(np.dot(inputs, weight_matrix))
        activations.append(inputs)
    return activations

# Prediction function
def predict(X, weights):
    activations = forward_propagation(X, weights)
    return np.argmax(activations[-1], axis=1)

# Function to visualize samples along with predictions and true labels
def visualize_samples(X, y_true, y_pred, num_samples=10):
    # Get indices for all classes
    class_indices = [np.where(y_true == i)[0] for i in range(10)]
    
    # Randomly select indices ensuring we have a mix of different digits
    selected_indices = []
    for _ in range(num_samples):
        # Randomly choose a class that still has samples
        available_classes = [i for i, indices in enumerate(class_indices) if len(indices) > 0]
        if not available_classes:
            break
        chosen_class = np.random.choice(available_classes)
        
        # Randomly select an index from the chosen class
        selected_index = np.random.choice(class_indices[chosen_class])
        selected_indices.append(selected_index)
        
        # Remove the selected index from the class_indices
        class_indices[chosen_class] = class_indices[chosen_class][class_indices[chosen_class] != selected_index]

    # Create a grid of subplots
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        ax.imshow(X[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f"True: {y_true[idx]}, Pred: {y_pred[idx]}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Evaluation function with confusion matrix
def evaluate_model(X_test, y_test, weights, visualize=False):
    y_pred = predict(X_test, weights)
    
    correct_predictions = np.sum(y_pred == y_test)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions
    
    print(f"{datetime.now()} - Correct Predictions: {correct_predictions}/{total_predictions}")
    print(f"{datetime.now()} - Test Accuracy: {accuracy:.2f}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    if visualize:
        visualize_samples(X_test, y_test, y_pred, num_samples=10)

# Function to load weights
def load_weights(format='npy', prefix="weights_layer_"):
    if format not in ['npy', 'txt']:
        raise ValueError("Format must be either 'npy' or 'txt'")
    
    weights = []
    i = 0
    while True:
        try:
            if format == 'npy':
                filename = f"{prefix}{i}.npy"
                weight_matrix = np.load(filename)
            else:  # txt format
                filename = f"{prefix}{i}.txt"
                weight_matrix = np.loadtxt(filename, delimiter=",")
            
            weights.append(weight_matrix)
            i += 1
        except FileNotFoundError:
            break
    
    print(f"{datetime.now()} - Weights loaded from {format} files for {len(weights)} layers.")
    return weights

# Main function
def main(visualize=False, weight_format='npy', use_mnist=False):
    if use_mnist:
        mnist_file = "normalized_mnist/mnist_test_normalized.csv"
        X, y = load_normalized_mnist(mnist_file)
    else:
        # Load the normalized synthetic data
        output_dir = "normalized_digits"  # Directory where normalized images were saved
        X, y = load_data_from_csv(output_dir)

    # Load weights
    weights = load_weights(format=weight_format)
    print(f"{datetime.now()} - Loaded weights from {weight_format} files.")

    # Print shapes of loaded weights
    for i, weight_matrix in enumerate(weights):
        print(f"Weight matrix {i} shape: {weight_matrix.shape}")

    # Evaluate the model
    print(f"{datetime.now()} - Evaluating model on {'normalized MNIST' if use_mnist else 'synthetic'} data:")
    evaluate_model(X, y, weights, visualize=visualize)

# Set visualize flag, weight format, and dataset choice here
if __name__ == "__main__":
    main(visualize=True, weight_format='npy', use_mnist=True)  # Set use_mnist=False for synthetic data
    main(visualize=True, weight_format='npy', use_mnist=False)  # Set use_mnist=False for synthetic data