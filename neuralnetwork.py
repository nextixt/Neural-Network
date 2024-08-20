import numpy as np
import random
def sigmoid(x):
    return 1 / (1 + 2.71828182845904 ** -x)
input1 = np.array([1,0,1,0])
input2 = np.array([0,1,0,1])

for i in range(1000):
    weight1 = round(random.uniform(0, 1), 1)
    weight2 = round(random.uniform(0, 1), 1)
output = sigmoid(input1 * weight1 + input2 * weight2)
print(output)
