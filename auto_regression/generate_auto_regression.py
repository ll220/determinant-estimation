import numpy as np
import time
from statistics import mean
import matplotlib.pyplot as plt
from operator import itemgetter

CHANGE_RANGE = 100
TOTAL_VALS = 2 * CHANGE_RANGE

A = np.array([[0.0, 0.0], [0.0, 0.0]])  # A[0][0] + A[1][0] < 1 seems to work best
                                        # same with A[0][1] + A[1][1]

C = np.array([[0.0], [0.0]])

Y1 = np.array([[0.0], [0.0]])


COV1 = [[1.0, 0], [0, 1.0]]
COV2 = [[30.0, 0], [0, 30.0]]
RANDOM_MEAN = [0, 0]



def generate_curr_y(cov, prev_y): 
    e1, e2 = np.random.multivariate_normal(RANDOM_MEAN, cov)

    error_val = np.array([[e1], [e2]])

    curr_y = np.matmul(A, prev_y) + C + error_val

    return curr_y, error_val

def generate_autoregression():
    curr_y = Y1
    y_values = []
    error_vals = []

    for x in range(CHANGE_RANGE):
        curr_y, error_val = generate_curr_y(COV1, curr_y)
        y_values.append(curr_y)
        error_vals.append(error_val)

    for x in range(CHANGE_RANGE):
        curr_y, error_val = generate_curr_y(COV2, curr_y)
        y_values.append(curr_y)
        error_vals.append(error_val)

    y1_values = []
    y2_values = []

    for y in y_values:
        y1_values.append(y[0][0])
        y2_values.append(y[1][0])

    return y1_values, y2_values, error_vals


def calculate_S(error_vals, h):
    s_matrix = np.array([[0, 0], [0, 0]])
    s1_matrix = np.array([[0, 0], [0, 0]])
    s2_matrix = np.array([[0, 0], [0, 0]])

    for i in range(0, h):
        error_val = error_vals[i]
        numerator = np.matmul(error_val, np.transpose(error_val))

        s1_total = np.divide(numerator, h)
        s_total = np.divide(numerator, TOTAL_VALS)

        s_matrix = s_matrix + s_total
        s1_matrix = s1_matrix + s1_total
        
    (sign1, logdet1) = np.linalg.slogdet(s1_matrix)
    s1 = sign1 * logdet1

    for i in range(h, TOTAL_VALS):
        error_val = error_vals[i]
        numerator = np.matmul(error_val, np.transpose(error_val))

        s2_total = np.divide(numerator, TOTAL_VALS - h)
        s_total = np.divide(numerator, TOTAL_VALS)

        s_matrix = s_matrix + s_total
        s2_matrix = s2_matrix + s2_total

    # print(s_matrix)
    # print(s1_matrix)
    # print(s2_matrix)

    (sign2, logdet2) = np.linalg.slogdet(s2_matrix)
    s2 = sign2 * logdet2
    (sign_s, logdet_s) = np.linalg.slogdet(s_matrix)
    s = sign_s * logdet_s

    return s, s1, s2

def calculate_statistics(error_vals):
    statistics = []

    for i in range(1, TOTAL_VALS - 1):
        s, s1, s2 = calculate_S(error_vals, i)
        v = float(i / TOTAL_VALS)

        new_stat = TOTAL_VALS * (s - ((v*s1) + ((1-v)*(s2))))
        statistics.append(new_stat)

    return statistics
    

    


y1_values, y2_values, error_vals = generate_autoregression()

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

statistics = calculate_statistics(error_vals)

plt.plot(statistics)
plt.title("LRT Statistics")
plt.xlabel('Times - 2')
plt.ylabel('LRT values')
plt.show()




