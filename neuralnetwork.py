import random
import numpy as np
def sigmoid(sum):
    return 1 / (1 + np.exp(-sum))
X = np.array([
    [0,1,1,0],
    [1,1,0,1],
    [0,1,0,0],
    [1,1,0,0]
    ])

y = np.array([0,1,1,0])
b = 0.1
w = np.random.uniform(-1, 1, size=X.shape[1])
learning_rate = 0.01 

for i in range(200):
    y_pred = sigmoid(w @ X + b)

    loss = (y_pred - y) ** 2

    w = w - learning_rate * loss @ X
    b = b - learning_rate * loss
print(f"Loss: {sum(loss)}")
print(f"Prediction: {y_pred}")
print(f"Current w: {w}, Current b: {b}")
