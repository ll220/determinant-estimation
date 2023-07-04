import numpy as np
import time
from statistics import mean
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy import stats
import math


LOGGING = False
DIM = 5
N = 5

tridiag_matrix = []


RANDOM_MEAN = 0
RANDOM_ST = 10


def set_up_a_matrix(size):
    triangle_matrix = generate_lower_triangle_matrix(size)
    transpose = triangle_matrix.transpose()
    
    pos_semi_def_matrix = np.matmul(triangle_matrix, transpose)

    # print("Random Triangle matrix: \n", triangle_matrix)
    # print("\nTranspose: \n", transpose)
    # print("\n\n Random Positive Semi-Definite matrix size ", size, ": \n", pos_semi_def_matrix)
    return pos_semi_def_matrix

def generate_lower_triangle_matrix(size):
    triangle_matrix = np.zeros(shape=(size, size))
    for x in range(size):
        for y in range(size - x):
            triangle_matrix[size - x - 1][y] = np.random.normal(loc=RANDOM_MEAN, scale=RANDOM_ST, size=None)

            if (size - x - 1) == y and triangle_matrix[size - x - 1][y] < 0.0:
                triangle_matrix[size - x - 1][y] = 1.0

    return triangle_matrix


def generate_b_vector_q1():
    b_vector = np.random.randint(0, 10, size=(DIM, 1))

    b_magnitude = 0
    for i in range(DIM):
        b_magnitude += pow(b_vector[i][0], 2)

    b_magnitude = math.sqrt(b_magnitude)
    q1 = np.divide(b_vector, b_magnitude)

    if(LOGGING):
        print("b vector: ")
        print(b_vector)
        print("\n")
    return b_vector, q1


def lanczos_iteration(a, q1):
    q_matrix = q1

    q_n_minus_1 = np.array([[0] * DIM])
    q_n_minus_1 = q_n_minus_1.transpose()

    qn = q1
    beta_n_minus_1 = 0
    beta_n = 0

    for n in range(1, N + 1):
        if(LOGGING):
            print("iteration n = ", str(n))

        v = np.matmul(a, qn)
        alpha_n = (np.matmul(qn.transpose(), v))[0][0]
        v = np.array(v) - (q_n_minus_1 * beta_n_minus_1) - (qn * alpha_n)
        beta_n = math.sqrt(sum(pow(i, 2) for i in v))

        if(LOGGING):
            print("alpha n: ", str(alpha_n))
            print("beta n: ", str(beta_n))
            print("beta n - 1: ", str(beta_n_minus_1))
            print("qn: ")
            print(qn)
            print("\n")

        append_tridiag_matrix(n, alpha_n, beta_n, beta_n_minus_1)

        if (n < N):
            q_n_plus_1 = np.divide(v, beta_n)
            q_n_minus_1 = qn
            qn = q_n_plus_1

            q_matrix = np.concatenate((q_matrix, qn), axis=1)

            beta_n_minus_1 = beta_n

    return q_matrix


def append_tridiag_matrix(n, alpha_n, beta_n, beta_n_minus_1):
    row = []

    if (n == 1):
        row.append(alpha_n)
        row.append(beta_n)
        last_half = [0] * (DIM - 2)
        row = row + last_half

    elif (n == DIM):
        first_half = [0] * (DIM - 2)
        row = row + first_half
        row.append(beta_n_minus_1)
        row.append(alpha_n)

    else:
        first_half = [0] * (n - 2)
        row = row + first_half
        row.append(beta_n_minus_1)
        row.append(alpha_n)
        row.append(beta_n)
        second_half = [0] * (DIM - n - 1)
        row = row + second_half

    tridiag_matrix.append(row)

def calculate_determinant(tridiag_matrix):
    None

b_vector, q1 = generate_b_vector_q1()
q_matrix = q1
a = set_up_a_matrix(DIM)

print(a)

a_log = np.log(a)
print("a log:")
print(a_log)

q_matrix = lanczos_iteration(a, q1)

finished_tridiag = np.array(tridiag_matrix)

if(LOGGING):
    print("q matrix: ")
    print(q_matrix)
    print("\n")

    print("tridiagonal matrix: ")
    print(finished_tridiag)
    print("\n")

sub_tridiag_matrix = finished_tridiag[0:N, 0:N]

if(LOGGING):
    print("sub tridiagonal matrix: ")
    print(sub_tridiag_matrix)
    print("\n")


if(LOGGING):
    test_matrix = np.matmul(q_matrix.transpose(), a)
    test_matrix = np.matmul(test_matrix, q_matrix)

if(LOGGING):
    print("test matrix: ")
    print(test_matrix)
    print("\n")

print(np.linalg.det(a))
print(np.linalg.det(sub_tridiag_matrix))
