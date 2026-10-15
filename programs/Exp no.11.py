import numpy as np
import matplotlib.pyplot as plt

# Create sample image
image = np.array([
    [1,1,1,0,0],
    [1,1,1,0,0],
    [1,1,1,0,0],
    [0,0,0,1,1],
    [0,0,0,1,1]
])

# Simple filter (kernel)
kernel = np.array([
    [1,0],
    [0,-1]
])

# Convolution
output = np.zeros((4,4))

for i in range(4):
    for j in range(4):
        region = image[i:i+2, j:j+2]
        output[i,j] = np.sum(region * kernel)

print("Convolution Output:")
print(output)

plt.imshow(output, cmap='gray')
plt.title("CNN Feature Map")
plt.show()