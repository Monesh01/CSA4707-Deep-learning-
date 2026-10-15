import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sample Dataset
data = {
    'X': [2, 4, 6, 8, 10],
    'Y': [1, 3, 5, 7, 9]
}

df = pd.DataFrame(data)

# Covariance Matrix
cov_matrix = np.cov(df['X'], df['Y'])
print("Covariance Matrix:\n")
print(cov_matrix)

# Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:\n")
print(eigenvalues)

print("\nEigenvectors:\n")
print(eigenvectors)

# Scatter Plot
plt.scatter(df['X'], df['Y'], color='blue')
plt.title("Scatter Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
        