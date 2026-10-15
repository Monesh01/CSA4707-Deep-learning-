import numpy as np

# Input values
inputs = np.array([1, 0])

# Weights
weights = np.array([0.5, 0.5])

# Bias
bias = -0.2

# Weighted Sum
net_input = np.dot(inputs, weights) + bias

# Step Activation Function
if net_input >= 0:
    output = 1
else:
    output = 0

print("Inputs:", inputs)
print("Weights:", weights)
print("Bias:", bias)
print("Net Input:", net_input)
print("Output:", output)