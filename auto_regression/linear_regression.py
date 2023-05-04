import numpy as np

A1 = 1
A2 = 3
A0 = 2

NUM_TRIALS = 1000

def generate_x():
    x1_values = []
    x2_values = []
    for x in range(0, NUM_TRIALS):
        x1_values.append(x)

    for x in range(NUM_TRIALS, 0, -1):
        x2_values.append(x)

    return np.array(x1_values), np.array(x2_values)

def calculate_y(x1_values, x2_values):
    y_values = []
    for i in range(0, NUM_TRIALS):
        x1 = x1_values[i]
        x2 = x2_values[i]

        epsilon = np.random.normal(loc=0, scale=5)
        curr_y = A1 * x1 + A2 * x2 + A0 + epsilon
        y_values.append(curr_y)

    return np.array(y_values)

def calculate_a(x1_values, x2_values, y_values):
    x_transpose = np.array([x1_values, x2_values])
    x_matrix = x_transpose.transpose()

    first_bit = np.linalg.inv(np.matmul(x_transpose, x_matrix))
    second_bit = np.matmul(first_bit, x_transpose)
    a_matrix = np.matmul(second_bit, y_values)
    return a_matrix

x1_values, x2_values = generate_x()
y_values = calculate_y(x1_values, x2_values)
a_matrix = calculate_a(x1_values, x2_values, y_values)
print(a_matrix)