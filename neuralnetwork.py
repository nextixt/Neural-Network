import random
def sigmoid(sum):
    return 1 / (1 + 2.7183 ** -sum)
X = -1
y = 1
b = 0.1
w = random.uniform(-1,1)
learning_rate = 0.01 

for i in range(200):
    y_pred = sigmoid(w * X + b)

    loss = (y_pred - y) ** 2

    w = w - learning_rate * loss * X
    b = b - learning_rate * loss
print(loss)

y_pred = sigmoid(w * X + b)
print(f"Prediction: {y_pred}")
print(f"Current w: {w}, Current b: {b}")
