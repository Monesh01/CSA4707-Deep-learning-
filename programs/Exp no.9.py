import numpy as np

# Input and Output
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Weights Initialization
np.random.seed(1)

W1 = np.random.randn(2,4)
W2 = np.random.randn(4,1)

# Sigmoid Function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative(x):
    return x*(1-x)

# Training
for i in range(5000):

    hidden = sigmoid(np.dot(X,W1))
    output = sigmoid(np.dot(hidden,W2))

    error = y - output

    d_output = error * derivative(output)
    d_hidden = d_output.dot(W2.T) * derivative(hidden)

    W2 += hidden.T.dot(d_output)*0.1
    W1 += X.T.dot(d_hidden)*0.1

print("Output:")
print(output)