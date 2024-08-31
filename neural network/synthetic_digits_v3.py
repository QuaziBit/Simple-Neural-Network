from PIL import Image, ImageDraw, ImageFont
import os
import random
import numpy as np

def generate_digit_image(digit, font_size, font_path, base_dir, augment=False, add_noise=False):
    # Create a larger blank black image (56x56)
    image = Image.new('L', (56, 56), color=0)  # 'L' mode is for grayscale, 0 is black
    
    # Create a drawing context
    draw = ImageDraw.Draw(image)
    
    # Load a font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print(f"Font IOError: {font_path}")
    
    # Get the size of the digit
    text_size = draw.textsize(str(digit), font=font)
    
    # Calculate the position to center the digit in the larger image
    text_x = (56 - text_size[0]) // 2
    text_y = (56 - text_size[1]) // 2
    
    # Draw the digit in white
    draw.text((text_x, text_y), str(digit), font=font, fill=255)
    
    # Apply augmentation if enabled
    if augment:
        image = augment_image(image)
    
    # Crop the image to 28x28 from the center
    left = (56 - 28) // 2
    top = (56 - 28) // 2
    right = left + 28
    bottom = top + 28
    image = image.crop((left, top, right, bottom))
    
    # Apply noise if requested
    if add_noise:
        image = add_noise_to_image(image)
    
    # Determine the suffix for the directory name
    noise_suffix = "_noisy" if add_noise else ""
    digit_dir = os.path.join(base_dir, str(digit) + noise_suffix)
    
    if not os.path.exists(digit_dir):
        os.makedirs(digit_dir)
    
    image_filename = f"{digit}_{random.randint(0, 9999)}.png"
    image.save(os.path.join(digit_dir, image_filename))

def augment_image(image):
    # Random rotation (reduced range to keep digit more centered)
    angle = random.uniform(-5, 5)
    image = image.rotate(angle, fillcolor=0, center=(28, 28))
    
    # Random translation (reduced range)
    translate_x = random.randint(-2, 2)
    translate_y = random.randint(-2, 2)
    image = image.transform(image.size, Image.AFFINE, (1, 0, translate_x, 0, 1, translate_y), fillcolor=0)
    
    return image

def add_noise_to_image(image):
    np_image = np.array(image)
    noise = np.random.normal(0, 10, np_image.shape)  # Reduced noise intensity
    np_image = np.clip(np_image + noise, 0, 255)
    return Image.fromarray(np_image.astype('uint8'))

# Main execution
base_dir = "synthetic_digits"
font_paths = [
    "./fonts/file-name-1.ttf",
    "./fonts/file-name-2.ttf",
    "./fonts/file-name-3.ttf",
    "./fonts/file-name-4.ttf",
    "./fonts/file-name-5.ttf"
]
digits = list(range(10))  # Digits 0 to 9
font_sizes = [20, 22, 24]  # Increased font sizes for better visibility

total_images = 10000
images_per_digit = total_images // 10

for digit in digits:
    for _ in range(images_per_digit):
        font_path = random.choice(font_paths)
        font_size = random.choice(font_sizes)
        
        # Generate clear image
        generate_digit_image(digit, font_size, font_path, base_dir, augment=True, add_noise=False)
        
        # Generate noisy image
        generate_digit_image(digit, font_size, font_path, base_dir, augment=True, add_noise=True)

print(f"Generated {total_images * 2} images in total.")