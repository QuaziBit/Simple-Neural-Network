import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog
import seaborn as sns
from PIL import Image, ImageDraw, ImageTk, ImageOps

# Assuming you have these functions from your previous scripts
from simple_neural_network_v3 import forward_propagation, leaky_relu, softmax
from test_model_v3 import load_data_from_csv

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
    
    print(f"Weights loaded from {format} files for {len(weights)} layers.")
    return weights

def prepare_image_for_network(image, decimal_places=5):
    """
    Prepare the image for the neural network by converting it to grayscale,
    normalizing, and flattening the array.
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Ensure the image is 28x28
    if img_array.shape != (28, 28):
        raise ValueError(f"Input image must be 28x28 pixels, got {img_array.shape}")
    
    # Normalize the image to range 0.01 to 1.00
    normalized_image = (img_array / 255.0 * 0.99) + 0.01
    
    # Round the normalized values to the specified number of decimal places
    rounded_image = np.round(normalized_image, decimals=decimal_places)
    
    return rounded_image.flatten()

class DrawingApp:
    def __init__(self, master, update_callback, decimal_places):
        self.master = master
        self.update_callback = update_callback
        self.decimal_places = decimal_places
        self.setup_ui()
        self.reset_image()

    def setup_ui(self):
        self.canvas = tk.Canvas(self.master, width=280, height=280, bg="black", cursor="cross")
        self.canvas.pack(side=tk.LEFT, padx=5, pady=5)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        button_frame = ttk.Frame(self.master)
        button_frame.pack(side=tk.LEFT, padx=5, pady=5)

        clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_canvas)
        clear_button.pack(side=tk.TOP, padx=5, pady=5)

        recognize_button = ttk.Button(button_frame, text="Recognize", command=self.recognize)
        recognize_button.pack(side=tk.TOP, padx=5, pady=5)

        upload_button = ttk.Button(button_frame, text="Upload Image", command=self.upload_image)
        upload_button.pack(side=tk.TOP, padx=5, pady=5)

        self.debug_label = tk.Label(self.master, text="Debug: Not drawn yet")
        self.debug_label.pack(side=tk.BOTTOM, padx=5, pady=5)

    def reset_image(self):
        self.image = Image.new("RGB", (280, 280), color="black")
        self.draw = ImageDraw.Draw(self.image)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.draw.line([x1, y1, x2, y2], fill="white", width=5)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.debug_label.config(text=f"Debug: Drawing at ({x1}, {y1})")

    def reset(self, event):
        self.debug_label.config(text="Debug: Drawing ended")

    def clear_canvas(self):
        self.reset_image()
        self.debug_label.config(text="Debug: Canvas cleared")

    def recognize(self):
        gray_image = self.image.convert("L")
        resized_image = gray_image.resize((28, 28), Image.Resampling.LANCZOS)
        processed_img = prepare_image_for_network(resized_image, self.decimal_places)
        self.update_callback(processed_img)
        self.debug_label.config(text="Debug: Recognition triggered")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if file_path:
            uploaded_image = Image.open(file_path).convert("L")
            uploaded_image = uploaded_image.resize((28, 28), Image.Resampling.LANCZOS)
            processed_img = prepare_image_for_network(uploaded_image, self.decimal_places)
            display_image = Image.fromarray((processed_img.reshape(28, 28) * 255).astype('uint8'))
            display_image = display_image.resize((280, 280), Image.Resampling.NEAREST)
            self.image = display_image.convert("RGB")
            self.draw = ImageDraw.Draw(self.image)
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
            self.debug_label.config(text="Debug: Image uploaded and preprocessed")
            self.update_callback(processed_img)

def visualize_network(fig, digit, weights, activations):
    fig.clear()
    num_layers = len(weights) + 1
    gs = fig.add_gridspec(3, num_layers)
    
    # Input image
    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.imshow(digit.reshape(28, 28), cmap='gray')
    ax_input.set_title("Input Image")
    ax_input.axis('off')
    
    # Weights visualization for each layer
    for i in range(num_layers - 1):
        ax_weights = fig.add_subplot(gs[0, i+1])
        
        if i == num_layers - 2:  # Last layer
            predicted_digit = np.argmax(activations[-1])
            relevant_weights = weights[i][:, predicted_digit].reshape(-1, 1)
        else:
            # For hidden layers, visualize the average of absolute weights
            relevant_weights = np.mean(np.abs(weights[i]), axis=1).reshape(-1, 1)
        
        # Normalize weights for better visualization
        relevant_weights = (relevant_weights - relevant_weights.min()) / (relevant_weights.max() - relevant_weights.min())
        
        sns.heatmap(relevant_weights, ax=ax_weights, cmap='RdBu_r', center=0)
        ax_weights.set_title(f"Layer {i+1} Weights")
    
    # Output Layer Activations
    ax_output = fig.add_subplot(gs[1, :])
    sns.heatmap(activations[-1].reshape(1, -1), ax=ax_output, cmap='viridis', cbar=False, annot=True, fmt='.2f')
    ax_output.set_title("Output Layer Activations")
    ax_output.set_yticks([])
    ax_output.set_xticks(range(10))
    ax_output.set_xticklabels(range(10))
    
    # Network structure and activations
    ax_network = fig.add_subplot(gs[2, :])
    layer_sizes = [784] + [w.shape[1] for w in weights]
    
    max_neurons = max(layer_sizes)
    neuron_grid = np.zeros((len(layer_sizes), max_neurons))
    
    for i, (size, activation) in enumerate(zip(layer_sizes, activations)):
        neuron_grid[i, :size] = activation.flatten()[:size]
    
    sns.heatmap(neuron_grid, ax=ax_network, cmap='viridis', cbar_kws={'label': 'Activation Strength'}, vmin=0, vmax=1.5)
    
    ax_network.set_yticks(np.arange(len(layer_sizes)) + 0.5)
    ax_network.set_yticklabels([f'Layer {i} ({size})' for i, size in enumerate(layer_sizes)])
    ax_network.set_title("Neural Network Activations")
    ax_network.set_xlabel("Neurons")
    
    predicted_digit = np.argmax(activations[-1])
    fig.suptitle(f"Digit Recognition Process\nPredicted: {predicted_digit}", fontsize=16)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

def update_visualization(digit, weights):
    activations = forward_propagation(digit.reshape(1, -1), weights)
    visualize_network(fig, digit, weights, activations)
    canvas.draw()

def main(weight_format='npy'):
    global weights, fig, canvas

    weights = load_weights(format=weight_format)

    root = tk.Tk()
    root.title("Interactive Digit Recognition")

    # Add this line to define what happens when the window is closed
    root.protocol("WM_DELETE_WINDOW", lambda: root.quit())

    drawing_frame = ttk.Frame(root)
    drawing_frame.pack(side=tk.LEFT, padx=10, pady=10)

    viz_frame = ttk.Frame(root)
    viz_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    fig = plt.figure(figsize=(12, 8))
    canvas = FigureCanvasTkAgg(fig, master=viz_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    decimal_places = 5
    drawing_app = DrawingApp(drawing_frame, lambda digit: update_visualization(digit, weights), decimal_places)

    blank_input = np.zeros((28, 28))
    update_visualization(blank_input.flatten(), weights)

    root.mainloop()

if __name__ == "__main__":
    main(weight_format='npy')  # Change to 'txt' to use text files