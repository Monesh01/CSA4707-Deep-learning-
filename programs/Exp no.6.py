import numpy as np
import matplotlib.pyplot as plt

# Sample Data
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

# Parameters
m = 0
b = 0
learning_rate = 0.01
epochs = 100

losses = []

for i in range(epochs):
    y_pred = m * X + b

    dm = (-2 / len(X)) * np.sum(X * (y - y_pred))
    db = (-2 / len(X)) * np.sum(y - y_pred)

    m = m - learning_rate * dm
    b = b - learning_rate * db

    loss = np.mean((y - y_pred) ** 2)
    losses.append(loss)

print("Slope (m):", m)
print("Intercept (b):", b)

plt.plot(losses)
plt.title("Gradient Descent Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()