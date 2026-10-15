import numpy as np
import matplotlib.pyplot as plt

# Vector Operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

print("Vector 1:", v1)
print("Vector 2:", v2)

print("Vector Addition:", v1 + v2)
print("Vector Subtraction:", v1 - v2)
print("Dot Product:", np.dot(v1, v2))

# Matrix Operations
m1 = np.array([[1, 2],
               [3, 4]])

m2 = np.array([[5, 6],
               [7, 8]])

print("\nMatrix 1:\n", m1)
print("\nMatrix 2:\n", m2)

print("\nMatrix Addition:\n", m1 + m2)
print("\nMatrix Multiplication:\n", np.matmul(m1, m2))
print("\nTranspose of Matrix 1:\n", m1.T)

# Simple Visualization
plt.imshow(m1, cmap='viridis')
plt.colorbar()
plt.title("Matrix Visualization")
plt.show()


