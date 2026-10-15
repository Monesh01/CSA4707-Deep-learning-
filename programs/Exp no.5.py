import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Sample data
X = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
y = np.array([1.2,1.8,3.2,3.9,5.1,5.8,7.2,8.1,8.9,10.2])

# Underfitting Model (Degree 1)
underfit = make_pipeline(
    PolynomialFeatures(1),
    LinearRegression()
)

# Good Fit Model (Degree 2)
goodfit = make_pipeline(
    PolynomialFeatures(2),
    LinearRegression()
)

# Overfitting Model (Degree 9)
overfit = make_pipeline(
    PolynomialFeatures(9),
    LinearRegression()
)

# Train models
underfit.fit(X, y)
goodfit.fit(X, y)
overfit.fit(X, y)

# Predictions
X_test = np.linspace(1, 10, 100).reshape(-1,1)

# Plot
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='black', label='Data')

plt.plot(X_test, underfit.predict(X_test),
         label='Underfitting (Degree 1)', color='red')

plt.plot(X_test, goodfit.predict(X_test),
         label='Good Fit (Degree 2)', color='green')

plt.plot(X_test, overfit.predict(X_test),
         label='Overfitting (Degree 9)', color='blue')

plt.legend()
plt.title("Bias-Variance Analysis")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
    