import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Generate data for Normal Distribution
x = np.linspace(-5, 5, 100)
y = norm.pdf(x, 0, 1)

# Plot Normal Distribution
plt.plot(x, y)
plt.title("Normal Probability Distribution")
plt.xlabel("X")
plt.ylabel("Probability Density")
plt.grid(True)
plt.show()

# Bayes Theorem Example
P_A = 0.01      # Disease probability
P_B_A = 0.95    # Positive test given disease
P_B = 0.10      # Positive test probability

P_A_B = (P_B_A * P_A) / P_B

print("Probability using Bayes Theorem =", P_A_B)
