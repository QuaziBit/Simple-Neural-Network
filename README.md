# Simple-Neural-Network
A simple neural network for handwritten digits recognition - Python 3.

# MNIST data set
URL: https://pjreddie.com/projects/mnist-in-csv/

# Generate synthetic digits
run: synthetic_digits_v3.py \
run: preprocess_image_v3.py

## preprocess_image_v3
This script normalized our data sets, you have to run it for synthetic digits or mnist digits.

# Train Neural Network
run: simple_neural_network_v3.py \
This script should produce ".npy" or ".txt" layer's weights.

## test_model_v3.py
You can run test_model_v3.py if you want to get some testing of the current weights that you got after training.

# Test how Neural Network can recognize written digits
run: visualize_model_v3.py