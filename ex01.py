import numpy as np


# Neuron for logical OR
def logical_or_neuron(z):
    weights = np.ones_like(z)
    bias = -0.5
    pre_activation = np.dot(weights, z) + bias
    activation = 1 if pre_activation > 0 else 0
    return activation


# Neuron for masked logical OR
def masked_logical_or_neuron(z, c):
    weights = np.array(c)
    bias = -0.5
    pre_activation = np.dot(weights, z) + bias
    activation = 1 if pre_activation > 0 else 0
    return activation


# Neuron for perfect match
def perfect_match_neuron(z, c):
    activation = 1 if np.array_equal(z, c) else 0
    return activation


# Second Layer (Softmax Activation)
def softmax_layer(inputs):
    exp_inputs = np.exp(inputs)
    softmax_outputs = exp_inputs / np.sum(exp_inputs)
    return softmax_outputs


# Third Layer (One-Hot Encoding)
def one_hot_encoding(outputs):
    one_hot = np.zeros_like(outputs)
    one_hot[np.argmax(outputs)] = 1
    return one_hot


# # Example usage
# z = np.array([1, 1])  # Sample input
# c = np.array([1, 0])  # Sample binary vector
#
# # First Layer
# corner1 = logical_or_neuron(z)
# corner2 = masked_logical_or_neuron(z, c)
# corner3 = perfect_match_neuron(z, c)
#
# # Second Layer
# inputs = np.array([corner1, corner2, corner3])
# print(inputs)
# # softmax_outputs = softmax_layer(inputs)
#
# # Third Layer (Output Layer)
# output = one_hot_encoding(inputs)
#
# # Print the output of the network
# print(output)  # One-hot encoded output representing the class labels
