import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from tabulate import tabulate
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Function to load the normalized MNIST data from CSV files
def load_mnist_data(file_path):
    print(f"{datetime.now()} - Loading data from {file_path}")
    data = pd.read_csv(file_path, header=None)
    labels = data.iloc[:, 0].values
    pixels = data.iloc[:, 1:].values
    return pixels, labels

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

def visualize_network(weights, activations, epoch):
    fig, axes = plt.subplots(len(weights) + 1, 1, figsize=(12, 4 * (len(weights) + 1)))
    fig.suptitle(f"Network Visualization - Epoch {epoch}")

    # Visualize input
    ax = axes[0]
    im = ax.imshow(activations[0].reshape(-1, 28), aspect='auto', cmap='viridis')
    ax.set_title("Input Layer")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # Visualize weights and activations for each layer
    for i, (w, a) in enumerate(zip(weights, activations[1:])):
        ax = axes[i+1]
        im = ax.imshow(w, aspect='auto', cmap='coolwarm')
        ax.set_title(f"Layer {i+1} Weights")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.show()

# Activation functions
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# Forward propagation
def forward_propagation(inputs, weights):
    activations = [inputs]
    for weight_matrix in weights[:-1]:
        inputs = leaky_relu(np.dot(inputs, weight_matrix))
        activations.append(inputs)
    
    output_layer_output = softmax(np.dot(inputs, weights[-1]))
    activations.append(output_layer_output)
    
    return activations

# Backward propagation with L2 regularization
def backward_propagation(activations, expected_output, weights, learning_rate, lambda_reg=0.001):
    deltas = [activations[-1] - expected_output]
    
    for i in reversed(range(len(weights) - 1)):
        delta = np.dot(deltas[-1], weights[i + 1].T) * leaky_relu_derivative(activations[i + 1])
        deltas.append(delta)
    
    deltas.reverse()
    
    for i in range(len(weights)):
        gradient = np.dot(activations[i].T, deltas[i]) / activations[i].shape[0]
        gradient += lambda_reg * weights[i]  # L2 regularization
        weights[i] -= learning_rate * gradient
    
    return weights

# Function to print weights in a user-friendly format
def print_weight_summary(weights, epoch):
    summary = f"\nEpoch {epoch} - Weight summary: "
    for i, weight_matrix in enumerate(weights):
        avg = np.mean(weight_matrix)
        std = np.std(weight_matrix)
        min_val = np.min(weight_matrix)
        max_val = np.max(weight_matrix)
        summary += f"Layer {i+1} (shape {weight_matrix.shape}): "
        summary += f"Avg={avg:.4f}, Std={std:.4f}, Min={min_val:.4f}, Max={max_val:.4f} | "
    
    print(summary)

    # Print a sample of each weight matrix on separate lines
    for i, weight_matrix in enumerate(weights):
        sample_size = min(5, weight_matrix.shape[0])
        sample = weight_matrix[:sample_size, :sample_size]
        print(f"Layer {i+1} sample:")
        print(tabulate(sample, floatfmt=".4f", tablefmt="grid"))

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

# Training function with mini-batch gradient descent
def train_model(X_train, y_train, X_val, y_val, input_size, hidden_sizes, output_size, learning_rate=0.01, epochs=1000, batch_size=32, lambda_reg=0.001):
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    weights = [
        np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
        for i in range(len(layer_sizes) - 1)
    ]
    
    print(f"{datetime.now()} - Initializing weights and starting training.")
    print_weight_summary(weights, 0)  # Print initial weights
    
    num_samples = X_train.shape[0]
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_start_time = datetime.now()
        total_loss = 0.0
        
        permutation = np.random.permutation(num_samples)
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        for batch_start in range(0, num_samples, batch_size):
            batch_end = batch_start + batch_size
            X_batch = X_train_shuffled[batch_start:batch_end]
            y_batch = y_train_shuffled[batch_start:batch_end]
            
            activations = forward_propagation(X_batch, weights)
            batch_loss = -np.sum(y_batch * np.log(activations[-1] + 1e-8))
            total_loss += batch_loss
            
            weights = backward_propagation(activations, y_batch, weights, learning_rate, lambda_reg)
        
        average_loss = total_loss / num_samples
        train_losses.append(average_loss)
        
        # Compute validation loss
        val_activations = forward_propagation(X_val, weights)
        val_loss = -np.sum(y_val * np.log(val_activations[-1] + 1e-8)) / X_val.shape[0]
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"{datetime.now()} - Early stopping triggered at epoch {epoch+1}")
            break
        
        # Learning rate decay
        learning_rate *= 0.99
        
        epoch_end_time = datetime.now()
        
        # Print progress
        progress = (epoch + 1) / epochs * 100
        sys.stdout.write(f"\rEpoch {epoch+1}/{epochs} ({progress:.2f}%) - Train Loss: {average_loss:.4f}, Val Loss: {val_loss:.4f}")
        sys.stdout.flush()
        
        if (epoch + 1) % 100 == 0:
            print()  # New line for better readability
            print_weight_summary(weights, epoch+1)
            
            # Visualize network for a random sample
            sample_idx = np.random.randint(0, X_train.shape[0])
            sample_activations = forward_propagation(X_train[sample_idx:sample_idx+1], weights)
            visualize_network(weights, sample_activations, epoch+1)
    
    print(f"\n{datetime.now()} - Training completed.")
    return weights, train_losses, val_losses

# Functions to save and load weights
def save_weights(weights, format='npy', prefix="weights_layer_"):
    if format not in ['npy', 'txt']:
        raise ValueError("Format must be either 'npy' or 'txt'")
    
    for i, weight_matrix in enumerate(weights):
        if format == 'npy':
            filename = f"{prefix}{i}.npy"
            np.save(filename, weight_matrix)
        else:  # txt format
            filename = f"{prefix}{i}.txt"
            np.savetxt(filename, weight_matrix, delimiter=",")
    
    print(f"{datetime.now()} - Weights saved to {format} files for {len(weights)} layers.")


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
def main(visualize=False, use_mnist=True):
    if use_mnist:
        # Load the normalized MNIST data
        train_data_file = "normalized_mnist/mnist_train_normalized.csv"
        test_data_file = "normalized_mnist/mnist_test_normalized.csv"
        
        X_train, y_train = load_mnist_data(train_data_file)
        X_test, y_test = load_mnist_data(test_data_file)
        
        # Split training data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    else:
        # Load the synthetic data (keep existing code)
        output_dir = "normalized_digits"
        X, y = load_data_from_csv(output_dir)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    input_size = 784
    hidden_sizes = [512, 256, 128, 64, 32]  # You can now easily add or remove hidden layers
    output_size = 10

    # Convert y to one-hot encoding
    y_train = np.eye(output_size)[y_train]
    y_val = np.eye(output_size)[y_val]
    y_test = np.eye(output_size)[y_test]

    print(f"{datetime.now()} - Training set size: {len(X_train)}")
    print(f"{datetime.now()} - Validation set size: {len(X_val)}")
    print(f"{datetime.now()} - Test set size: {len(X_test)}")

    # Train the model
    weights, train_losses, val_losses = train_model(
        X_train, y_train, X_val, y_val, input_size, hidden_sizes, output_size,
        learning_rate=0.01, epochs=1000, batch_size=32, lambda_reg=0.001
    )

    # Save weights in both formats
    save_weights(weights, format='npy')
    save_weights(weights, format='txt')

    # Evaluate the model
    evaluate_model(X_test, np.argmax(y_test, axis=1), weights, visualize=visualize)
    
    # Plot training and validation loss
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main(visualize=True, use_mnist=True)  # Set use_mnist=False to use synthetic data