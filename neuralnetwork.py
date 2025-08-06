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
# generating matrix of weights
w = np.random.uniform(-1, 1, size=X.shape[1])
learning_rate = 0.01 

for i in range(200):
    y_pred = sigmoid(w @ X + b)
    # counting loss
    loss = np.mean((y_pred - y) ** 2)
    # counting gradients
    grad = X.T @ (loss * y_pred * (1 - y_pred))
    #updating weights and bias
    w -= learning_rate * grad
    b -= learning_rate * np.sum(loss * y_pred * (1 - y_pred))

print(f"Loss: {loss}")
print(f"Prediction: {y_pred}")
print(f"Current w: {w}, Current b: {b}")
