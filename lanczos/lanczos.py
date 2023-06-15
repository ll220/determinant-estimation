import numpy as np
import time
from statistics import mean
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy import stats
import math

DIM = 3

tridiag_matrix = []


RANDOM_MEAN = 0
RANDOM_ST = 10

def set_up_a_matrix():
    a = np.random.randint(0, 10,size=(DIM,DIM))
    a_symm = (a + a.transpose())/2

    print("a matrix: ")
    print(a_symm)
    print("\n")
    return a_symm

def generate_b_vector_q1():
    b_vector = np.random.randint(0, 10, size=(DIM, 1))

    b_magnitude = 0
    for i in range(DIM):
        b_magnitude += pow(b_vector[i][0], 2)

    b_magnitude = math.sqrt(b_magnitude)
    q1 = np.divide(b_vector, b_magnitude)

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

    for n in range(1, DIM + 1):
        print("iteration n = ", str(n))
        v = np.matmul(a, qn)
        alpha_n = (np.matmul(qn.transpose(), v))[0][0]
        v = np.array(v) - (q_n_minus_1 * beta_n_minus_1) - (qn * alpha_n)
        beta_n = math.sqrt(sum(pow(i, 2) for i in v))

        print("alpha n: ", str(alpha_n))
        print("beta n: ", str(beta_n))
        print("beta n - 1: ", str(beta_n_minus_1))
        print("qn: ")
        print(qn)
        print("\n")

        append_tridiag_matrix(n, alpha_n, beta_n, beta_n_minus_1)

        if (n != DIM):
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

b_vector, q1 = generate_b_vector_q1()
q_matrix = q1
a = set_up_a_matrix()

q_matrix = lanczos_iteration(a, q1)

finished_tridiag = np.array(tridiag_matrix)

print("q matrix: ")
print(q_matrix)
print("\n")

print("tridiagonal matrix: ")
print(finished_tridiag)
print("\n")

test_matrix = np.matmul(np.linalg.inv(q_matrix), a)
test_matrix = np.matmul(test_matrix, q_matrix)

print("test matrix: ")
print(test_matrix)
print("\n")

print(np.linalg.det(a))
print(np.linalg.det(finished_tridiag))
