import numpy as np

# Input Dataset (XOR Problem)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Target Output
y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize Weights
np.random.seed(1)

input_neurons = 2
hidden_neurons = 3
output_neurons = 1

W1 = np.random.rand(input_neurons, hidden_neurons)
W2 = np.random.rand(hidden_neurons, output_neurons)

# Sigmoid Function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Training
learning_rate = 0.5

for epoch in range(5000):

    # Forward Propagation
    hidden_input = np.dot(X, W1)
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2)
    predicted_output = sigmoid(final_input)

    # Error Calculation
    error = y - predicted_output

    # Backpropagation
    d_output = error * sigmoid_derivative(predicted_output)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Weight Update
    W2 += hidden_output.T.dot(d_output) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate

# Output
print("Predicted Output:")
print(predicted_output)