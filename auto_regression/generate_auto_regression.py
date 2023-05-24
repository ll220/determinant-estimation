import numpy as np
import time
from statistics import mean
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy import stats

CHANGE_RANGE = 100
TOTAL_VALS = 2 * CHANGE_RANGE

A = [0.9, 0.1]  # A[0][0] + A[1][0] < 1 seems to work best
                                        # same with A[0][1] + A[1][1]

C = [0.1, 0.9]

Y1 = [0.0, 0.0]


COV1 = [[1.0, 0], [0, 1.0]]
COV2 = [[30.0, 0], [0, 30.0]]
RANDOM_MEAN = [0, 0]



def generate_curr_y(cov, prev_y1, prev_y2): 
    e1, e2 = np.random.multivariate_normal(RANDOM_MEAN, cov)

    error_val = np.array([[e1], [e2]])

    y1 = A[0] * prev_y1 + C[0] + e1
    y2 = A[1] * prev_y2 + C[1] + e2

    return y1, y2, error_val

def generate_autoregression():
    curr_y1 = Y1[0]
    curr_y2 = Y1[1]
    y1_values = []
    y2_values = []
    error_vals = []

    for x in range(CHANGE_RANGE):
        curr_y1, curr_y2, error_val = generate_curr_y(COV1, curr_y1, curr_y2)
        y1_values.append(curr_y1)
        y2_values.append(curr_y2)

        error_vals.append(error_val)

    for x in range(CHANGE_RANGE):
        curr_y1, curr_y2, error_val = generate_curr_y(COV2, curr_y1, curr_y2)
        y1_values.append(curr_y1)
        y2_values.append(curr_y2)

        error_vals.append(error_val)


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

    for i in range(h, TOTAL_VALS - 2):
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

    for i in range(1, TOTAL_VALS - 2):
        s, s1, s2 = calculate_S(error_vals, i)
        v = float(i / TOTAL_VALS)

        new_stat = TOTAL_VALS * (s - ((v*s1) + ((1-v)*(s2))))
        statistics.append(new_stat)

    return statistics
    
def plot_values(values, title, times_label, values_label):
    plt.plot(values)
    plt.title(title)
    plt.xlabel(times_label)
    plt.ylabel(values_label)
    plt.show()

def generate_linear_regression_matrix(values):
    x_matrix = []
    y_matrix = []

    for i in range(2, TOTAL_VALS):
        y_matrix.append(values[i])
        x_matrix.append([values[i-1], 1])

    return np.array(y_matrix), np.array(x_matrix)

def calculate_a(x_values, y_values):
    x_transpose = x_values.transpose()

    first_bit = np.linalg.inv(np.matmul(x_transpose, x_values))
    second_bit = np.matmul(first_bit, x_transpose)
    a_matrix = np.matmul(second_bit, y_values)
    return a_matrix 

def calculate_error(a1_matrix, a2_matrix, y1_values, y2_values):
    error_vals = []
    for i in range(1, TOTAL_VALS): 
        error_val1 = y1_values[i] - a1_matrix[0] * y1_values[i-1] - a1_matrix[1]
        error_val2 = y2_values[i] - a2_matrix[0] * y2_values[i-1] - a2_matrix[1]
        error_vals.append(np.array([[error_val1], [error_val2]]))

    return error_vals




y1_values, y2_values, error_vals = generate_autoregression()
# plot_values(y1_values, "Y1 values using autoregression", "Times", "Y1 values")
# plot_values(y2_values, "Y2 values using autoregression", "Times", "Y2 values")

y1_matrix, x1_matrix = generate_linear_regression_matrix(y1_values)
a1_matrix = calculate_a(x1_matrix, y1_matrix)
print(a1_matrix)

y2_matrix, x2_matrix = generate_linear_regression_matrix(y2_values)
a2_matrix = calculate_a(x2_matrix, y2_matrix)
print(a2_matrix)

recalculated_error = calculate_error(a1_matrix, a2_matrix, y1_values, y2_values)
estimate_statistics = calculate_statistics(recalculated_error)
plot_values(estimate_statistics, "Estimated LRT Statistics", "Times - 2", "Estimated LRT values")


for x in estimate_statistics:
    p_value = stats.chi2.cdf(x, 3)
    print("P_value for: ")
    print(p_value)

# statistics = calculate_statistics(error_vals)
# plot_values(statistics, "LRT Statistics", "Times - 2", "LRT values")





