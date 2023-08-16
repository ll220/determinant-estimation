import numpy as np
import timeit
from statistics import mean
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy import stats
import math
from scipy.linalg import eigh_tridiagonal



LANCZOS_LOGGING = False
VALUES_LOGGING = False
E_LOGGING = False

# NUM_V = 30
# DIM = 5 # dimensions of input matrix
# M = 5 # num of lanczos iterations

RANDOM_MEAN = 0
RANDOM_ST = 5

def set_up_a_matrix(dim):
    triangle_matrix = generate_lower_triangle_matrix(dim)
    transpose = triangle_matrix.transpose()
    
    pos_semi_def_matrix = np.matmul(triangle_matrix, transpose)

    # print("Random Triangle matrix: \n", triangle_matrix)
    # print("\nTranspose: \n", transpose)
    # print("\n\n Random Positive Semi-Definite matrix size ", size, ": \n", pos_semi_def_matrix)
    return pos_semi_def_matrix

def generate_lower_triangle_matrix(dim):
    triangle_matrix = np.zeros(shape=(dim, dim))
    for x in range(dim):
        for y in range(dim - x):
            triangle_matrix[dim - x - 1][y] = np.random.normal(loc=RANDOM_MEAN, scale=RANDOM_ST, size=None)

            if (dim - x - 1) == y and triangle_matrix[dim - x - 1][y] < 0.0:
                triangle_matrix[dim - x - 1][y] = 1.0

    return triangle_matrix


def generate_rademacher_vector_and_q1(dim):
    b_vector = np.random.choice([1, -1], p=[0.5, 0.5], size=(dim, 1))
    
    b_magnitude = (np.linalg.norm(b_vector, axis=0))[0]

    q1 = np.divide(b_vector, b_magnitude)

    return q1


def lanczos_iteration(dim, m, a, q1):
    tridiag_matrix = None
    q_n_minus_1 = np.zeros((dim, 1))
    q_matrix = q1
    qn = q1
    beta_n_minus_1 = 0
    beta_n = 0

    for n in range(1, m + 1):
        if(LANCZOS_LOGGING):
            print("iteration n = ", str(n))

        v = np.matmul(a, qn)
        alpha_n = (np.matmul(qn.transpose(), v))[0][0]
        v -= (q_n_minus_1 * beta_n_minus_1) - (alpha_n * qn)
        beta_n = (np.linalg.norm(v, axis=0))[0]

        if(LANCZOS_LOGGING):
            print("alpha n: ", str(alpha_n))
            print("beta n: ", str(beta_n))
            print("beta n - 1: ", str(beta_n_minus_1))
            print("qn: ")
            print(qn)
            print("\n")

        if (n == 1):
            tridiag_matrix = np.array([np.concatenate(([alpha_n, beta_n], np.zeros(dim - 2)))])
        elif(n == dim):
            row = np.array([np.concatenate((np.zeros(dim - 2), [beta_n_minus_1, alpha_n]))])
            tridiag_matrix = np.concatenate((tridiag_matrix, row), axis=0)

        else:
            row = np.array([np.concatenate((np.zeros(n - 2), [beta_n_minus_1, alpha_n, beta_n], np.zeros(dim - n - 1)))])
            tridiag_matrix = np.concatenate((tridiag_matrix, row), axis=0)
        
        if (n < m):
            q_n_minus_1 = qn
            qn = np.divide(v, beta_n)

            q_matrix = np.concatenate((q_matrix, qn), axis=1)

            beta_n_minus_1 = beta_n

    test_matrix = np.matmul(q_matrix.transpose(), a)
    test_matrix = np.matmul(test_matrix, q_matrix)
    print(test_matrix)
    return tridiag_matrix

# def append_tridiag_matrix(dim, tridiag_matrix, n, alpha_n, beta_n, beta_n_minus_1):
#     row = []

#     if (n == 1):
#         row.append(alpha_n)
#         row.append(beta_n)
#         last_half = [0] * (dim - 2)
#         row = row + last_half

#     elif (n == dim):
#         first_half = [0] * (dim - 2)
#         row = row + first_half
#         row.append(beta_n_minus_1)
#         row.append(alpha_n)

#     else:
#         first_half = [0] * (n - 2)
#         row = row + first_half
#         row.append(beta_n_minus_1)
#         row.append(alpha_n)
#         row.append(beta_n)
#         second_half = [0] * (dim - n - 1)
#         row = row + second_half

#     tridiag_matrix.append(row)
#     return tridiag_matrix

def estimate_determinant(num_v, dim, m):

    a = set_up_a_matrix(dim)

    # act_start_time = time.time()
    (sign, logabsdet) = np.linalg.slogdet(a)
    act_det = sign * logabsdet
    # act_end_time = time.time()

    # act_time = act_end_time - act_start_time

    # est_start_time = time.time()
    det_est = 0
    for i in range(num_v):
        q1 = generate_rademacher_vector_and_q1(dim)

        tridiag_matrix = lanczos_iteration(dim, m, a, q1)

        d = []
        e = []

        for n in range(m):
            d.append(tridiag_matrix[n][n])

        for n in range(m - 1):
            e.append(tridiag_matrix[n][n+1])

        evalues, evectors = eigh_tridiagonal(d, e)

        if(VALUES_LOGGING):
            print("a: ")
            print(a)
            print("q matrix: ")
            # print(q_matrix)
            print("\n")


            print("tridiagonal matrix: ")
            print(tridiag_matrix)
            print("\n")

            # print("sub tridiagonal matrix: ")
            # print(sub_tridiag_matrix)
            # print("\n")

            # test_matrix = np.matmul(q_matrix.transpose(), a)
            # test_matrix = np.matmul(test_matrix, q_matrix)

            print("test matrix: ")
            # print(test_matrix)
            print("\n")

        if (E_LOGGING):

            print("Eigenvalues of a: ")
            print(np.linalg.eigvalsh(a))

            print("\nEigenvalues of tridiag: ")
            print(evalues)
            print("\nEigenvectors of tridiag: ")
            print(evectors)

        for k in range(m):
            det_est = det_est + (evectors[k][0] * evectors[k][0] * np.log(evalues[k]))

    est_det = float(dim / num_v) * det_est
    # est_end_time = time.time()
    # est_time = est_end_time - est_start_time
    error = est_det - act_det
    return error
    # print(det_est)
    # print(sign, logabsdet)

# act_times = []
# est_times = []
# error_vals = []
# dims = []

# for x in range(5, 200, 5):
#     average = 0.0
#     for j in range(10):
#         error = estimate_determinant(30, x, x)
#         average += error
#         # act_times.append(act_time)
#         # est_times.append(est_time)

#     average /= 10.0
#     error_vals.append(average)
#     dims.append(x)


# plot_title = "Average Error vs. m Iterations with Dim 70"

# plt.plot(dims, error_vals)
# # plt.plot(dims, error_vals, label = "Standard Calc Times")
# # plt.plot(dims, est_times, label = "Lanczos Calc Times")
# # plt.legend()
# plt.title(plot_title)
# plt.xlabel('Iterations')
# plt.ylabel('Error')
# # plt.savefig('Increasing_iterations2.png')

# plt.show()

# times = []

# for i in range(100):
#     begin_time = timeit.default_timer()
#     input_vector, q1 = generate_rademacher_vector_and_q1(1000000)
#     times.append(timeit.default_timer() - begin_time)

# print(min(times))

a = set_up_a_matrix(5)
q1 = generate_rademacher_vector_and_q1(5)
tridiag_matrix = lanczos_iteration(5, 5, a, q1)

# print(tridiag_matrix)