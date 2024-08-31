import os
import numpy as np
import pandas as pd
import numpy as np
from PIL import Image

### preprocess synthetic data ###

# Function to load, preprocess, and save images
def load_and_preprocess_synthetic(base_dir, output_dir, decimal_places, include_noisy=False):
    # Loop over all digit subdirectories
    for digit in range(10):
        digit_dir = os.path.join(base_dir, f"{digit}")
        noisy_digit_dir = os.path.join(base_dir, f"{digit}_noisy")
        
        # Check if the directory exists
        if not os.path.exists(digit_dir):
            print(f"[ERROR] Directory does not exist: {digit_dir}")
            continue
        
        # Create the corresponding output directory
        output_digit_dir = os.path.join(output_dir, f"{digit}_normalized")
        os.makedirs(output_digit_dir, exist_ok=True)
        
        # Process regular images
        process_directory_images(digit_dir, output_digit_dir, decimal_places)

        # Process noisy images if specified
        if include_noisy and os.path.exists(noisy_digit_dir):
            process_directory_images(noisy_digit_dir, output_digit_dir, decimal_places, is_noisy=True)
    
# Function to process images in a directory
def process_directory_images(directory, output_directory, decimal_places, is_noisy=False):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            
            # Load the image
            original_image = Image.open(img_path).convert('L')
            
            # Normalize the image to range 0.01 to 1.00
            normalized_image = (np.array(original_image) / 255.0 * 0.99) + 0.01
            
            # Round the normalized values to the specified number of decimal places
            rounded_image = np.round(normalized_image, decimals=decimal_places)
            
            # Verify the normalization
            if np.any(rounded_image > 1) or np.any(rounded_image < 0.01):
                print(f"[*] Normalization error in {filename}: values out of bounds")
            else:
                print(f"[OK] Normalization successful for {filename}")
            
            # Modify filename for noisy images
            if is_noisy:
                filename = "noisy_" + filename
            
            # Save the normalized image as PNG
            save_png_path = os.path.join(output_directory, filename)
            save_normalized_image_as_png(rounded_image, save_png_path)
            
            # Save the normalized image as CSV with controlled precision
            save_csv_path = os.path.join(output_directory, filename.replace(".png", ".csv"))
            save_normalized_image_as_csv(rounded_image, save_csv_path, decimal_places)
            
            print(f"Processed and saved: {filename}")

# Save normalized image as PNG
def save_normalized_image_as_png(rounded_image, save_path):
    # Convert back to [0, 255] range for saving
    image_to_save = ((rounded_image - 0.01) / 0.99 * 255).astype(np.uint8)
    
    # Create an image from the array
    img = Image.fromarray(image_to_save)
    
    # Save the image
    img.save(save_path)

# Save normalized image as CSV with controlled precision
def save_normalized_image_as_csv(rounded_image, save_path, decimal_places):
    fmt = f"%.{decimal_places}f"
    
    # Convert each value to a string and truncate trailing zeros
    truncated_image = np.vectorize(lambda x: (f"%.{decimal_places}f" % x).rstrip('0').rstrip('.'))(rounded_image)
    
    # Save the truncated values as CSV
    np.savetxt(save_path, truncated_image, delimiter=",", fmt="%s")

### preprocess read data - mnist ###

def load_and_preprocess_mnist(file_path, output_dir, decimal_places):
    # Load the CSV file
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path, header=None)
    
    # Separate labels and pixel values
    labels = data.iloc[:, 0]
    pixels = data.iloc[:, 1:]
    
    # Normalize pixel values to range 0.01 to 1.00
    normalized_pixels = (pixels / 255.0 * 0.99) + 0.01
    
    # Round the normalized values to the specified number of decimal places
    rounded_pixels = np.round(normalized_pixels, decimals=decimal_places)
    
    # Verify the normalization
    if np.any(rounded_pixels > 1) or np.any(rounded_pixels < 0.01):
        print("[ERROR] Normalization error: values out of bounds")
    else:
        print("[OK] Normalization successful")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save normalized data
    output_file = os.path.join(output_dir, os.path.basename(file_path).replace('.csv', '_normalized.csv'))
    save_normalized_data(labels, rounded_pixels, output_file, decimal_places)
    
    print(f"Processed and saved: {output_file}")

def save_normalized_data(labels, pixels, save_path, decimal_places):
    # Convert each value to a string with the specified decimal places
    fmt = f"%.{decimal_places}f"
    truncated_pixels = np.vectorize(lambda x: fmt % x)(pixels)
    
    # Remove trailing zeros after the decimal point
    truncated_pixels = np.vectorize(lambda x: x.rstrip('0').rstrip('.') if '.' in x else x)(truncated_pixels)
    
    # Combine labels and truncated pixels
    combined_data = pd.concat([labels, pd.DataFrame(truncated_pixels)], axis=1)
    
    # Save the combined data as CSV
    combined_data.to_csv(save_path, header=False, index=False)

# mnist data
train_file = "real_data/mnist_train.csv"
test_file = "real_data/mnist_test.csv"
output_dir = "normalized_mnist"
decimal_places = 5

# Process training and test data
load_and_preprocess_mnist(train_file, output_dir, decimal_places)
load_and_preprocess_mnist(test_file, output_dir, decimal_places)

# synthetic images
base_dir = "synthetic_digits"  # Directory containing original images
output_dir = "normalized_digits"  # Directory where normalized images will be saved
include_noisy = True  # Set to True if you want to include noisy images

# Process all images in the synthetic_digits directory and save them in normalized_digits
load_and_preprocess_synthetic(base_dir, output_dir, decimal_places, include_noisy)
