import numpy as np
import time
from statistics import mean
import matplotlib.pyplot as plt
from operator import itemgetter

NUM_VALUES = 100

A = np.array([[0.2, 0.4], [0.2, 0.3]])  # A[0][0] + A[1][0] < 1 seems to work best
                                        # same with A[0][1] + A[1][1]

C = np.array([[2.0], [4.0]])

Y1 = np.array([[5.0], [50.0]])


SIGMA1 = 5.0
SIGMA2 = 20.0


def generate_curr_y(sigma, prev_y): 
    e1, e2 = generate_error(sigma)

    error_val = np.array([[e1], [e2]])

    curr_y = np.matmul(A, prev_y) + C + error_val

    return curr_y


def generate_error(sigma):
    e1 = np.random.normal(loc=0, scale=sigma, size=None)
    e2 = np.random.normal(loc=0, scale=sigma, size=None)

    return e1, e2

curr_y = Y1
y_values = []

for x in range(NUM_VALUES - 1):
    y_values.append(curr_y)
    curr_y = generate_curr_y(SIGMA1, curr_y)


for x in range(NUM_VALUES):
    y_values.append(curr_y)
    curr_y = generate_curr_y(SIGMA2, curr_y)

# y1_values = y_values[:, 0]
y1_values = []
y2_values = []

for y in y_values:
    y1_values.append(y[0][0])
    y2_values.append(y[1][0])

plt.plot(y1_values)
plt.title("Y1 values using autoregression")
plt.xlabel('Times')
plt.ylabel('Y1 values')
plt.show()


plt.plot(y2_values)
plt.title("Y2 values using autoregression")
plt.xlabel('Times')
plt.ylabel('Y2 values')
plt.show()