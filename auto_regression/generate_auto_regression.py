import numpy as np
import time
from statistics import mean
import matplotlib.pyplot as plt

NUM_VALUES = 100

A1 = 0.5
A2 = 20.0

C1 = 2.0
C2 = 4.0

FIRST_Y1 = 5.0
FIRST_Y2 = 50.0

SIGMA1 = 5.0
SIGMA2 = 20.0


def generate_curr_y(sigma, prev_y_one, prev_y_two): 
    e1, e2 = generate_error(sigma)

    curr_y1 = prev_y_one * A1 + C1 + e1
    curr_y2 = prev_y_two * A1 + C2 + e2

    return curr_y1, curr_y2


def generate_error(sigma):
    e1 = np.random.normal(loc=0, scale=sigma, size=None)
    e2 = np.random.normal(loc=0, scale=sigma, size=None)

    return e1, e2
 
curr_y1 = FIRST_Y1
curr_y2 = FIRST_Y2

y1_values = []
y2_values = []

for x in range(NUM_VALUES - 1):
    y1_values.append(curr_y1)
    y2_values.append(curr_y2)
    curr_y1, curr_y2 = generate_curr_y(SIGMA1, curr_y1, curr_y2)


for x in range(NUM_VALUES):
    y1_values.append(curr_y1)
    y2_values.append(curr_y2)
    curr_y1, curr_y2 = generate_curr_y(SIGMA2, curr_y1, curr_y2)



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