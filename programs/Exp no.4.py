import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Probability Distributions
P = np.array([0.4, 0.3, 0.2, 0.1])
Q = np.array([0.3, 0.3, 0.2, 0.2])

# Entropy
entropy_P = entropy(P, base=2)

# Cross Entropy
cross_entropy = -np.sum(P * np.log2(Q))

# KL Divergence
kl_divergence = np.sum(P * np.log2(P / Q))

print("Entropy =", entropy_P)
print("Cross Entropy =", cross_entropy)
print("KL Divergence =", kl_divergence)

# Visualization
labels = ['Entropy', 'Cross Entropy', 'KL Divergence']
values = [entropy_P, cross_entropy, kl_divergence]

plt.bar(labels, values, color=['blue', 'green', 'red'])
plt.title("Entropy, Cross Entropy and KL Divergence")
plt.ylabel("Value")
plt.show()
