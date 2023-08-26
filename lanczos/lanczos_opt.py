import numpy as np
import timeit
from statistics import mean
import matplotlib.pyplot as plt
import math
import scipy
from scipy.linalg.lapack import dstemr


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
    indices = np.triu_indices(dim)  # Get upper triangular indices

    values = np.random.normal(loc=RANDOM_MEAN, scale=RANDOM_ST, size=indices[0].shape)
    values[values < 0.0] = 1.0

    triangle_matrix[indices] = values

    return triangle_matrix


def generate_input_vector(dim):
    b_vector = np.random.choice([-1, 1], size=(dim, 1))
    q1 = b_vector / np.linalg.norm(b_vector)

    return q1


def lanczos_iteration(dim, m, a, q1):
    # Wait, could I actually port in a submatrix of a and q1 and only work with those? No that won't work, beta_n depends on the magnitude of v as a whole
    tridiag_matrix = np.zeros((m, dim))  # Initialize the tridiagonal matrix
    q_n_minus_1 = np.zeros((dim, 1))
    qn = np.copy(q1)

    beta_n_minus_1 = 0
    beta_n = 0

    for n in range(1, m + 1):
        v = np.dot(a, qn)
        alpha_n = np.dot(qn.transpose(), v)[0][0]
        v -= beta_n_minus_1 * q_n_minus_1 + alpha_n * qn
        beta_n = np.linalg.norm(v)
        

        if n == 1:
            tridiag_matrix[n - 1, 0:2] = [alpha_n, beta_n]
        elif n == dim:
            tridiag_matrix[n - 1, dim-2:] = [beta_n_minus_1, alpha_n]
        else:
            tridiag_matrix[n - 1, n - 2:n + 1] = [beta_n_minus_1, alpha_n, beta_n]

        if n < m:
            q_n_minus_1 = np.copy(qn)
            qn = v / beta_n
            beta_n_minus_1 = beta_n

    return tridiag_matrix

def estimate_determinant(a_matrix, num_v, dim, m):
    det_est = 0

    for i in range(num_v):
        q1 = generate_input_vector(dim)
        tridiag_matrix = lanczos_iteration(dim, m, a_matrix, q1)
        
        d = tridiag_matrix.diagonal()
        e = np.empty(m)
        e[0:m - 1] = tridiag_matrix.diagonal(-1)
        e[-1] = 0.0 
                
        # Call dstemr instead of eigh_tridiagonal
        _, evalues, evectors, _ = dstemr(d, e, len(d), 1, m, 1, m)
        det_est += np.sum(evectors[:, 0] ** 2 * np.log(evalues))
    
    est_det = float(dim / num_v) * det_est
    return est_det

# act_times = []
# est_times = []
# error_vals = []
# dims = []


# for dim in range(10, 1000, 50):    
#     average = 0.0
#     average_act_time = 0.0
#     average_est_time = 0.0
#     for j in range(10):

#         a_matrix = set_up_a_matrix(dim)

#         act_begin_time = timeit.default_timer()
#         (sign, logabsdet) = np.linalg.slogdet(a_matrix)
#         act_det = sign * logabsdet
#         act_time = (timeit.default_timer() - act_begin_time)

#         print(dim, logabsdet)

#         est_begin_time = timeit.default_timer()
#         est_determinant = estimate_determinant(a_matrix, 30, dim, dim)
#         est_time = (timeit.default_timer() - est_begin_time)

#         average += est_determinant - act_det
#         average_act_time += act_time
#         average_est_time += est_time


#     average /= 10.0
#     average_act_time /= 10.0
#     average_est_time /= 10.0

#     error_vals.append(average)
#     act_times.append(average_act_time)
#     est_times.append(average_est_time)
#     dims.append(dim)


# plot_title = "Average Error vs. Iterations with Dim=70"
# plt.xlabel('Iterations')
# plt.ylabel('Error')
# plt.plot(dims, error_vals)
# plt.savefig('Increasing_iterations.png')
# plt.show()

# plot_title = "Time vs. Dimensions for Determinant Estimations"
# plt.plot(dims, act_times, label = "Standard Calc Times")
# plt.plot(dims, est_times, label = "Lanczos Calc Times")
# plt.legend()
# plt.title(plot_title)
# plt.xlabel('Dimensions/Iterations')
# plt.ylabel('Time(sec)')
# plt.savefig('Time_updated.png')
# plt.show()

# times = []

# for i in range(100):
#     begin_time = timeit.default_timer()
#     input_vector, q1 = generate_rademacher_vector_and_q1(1000000)
#     times.append(timeit.default_timer() - begin_time)

# print(min(times))




a = set_up_a_matrix(1000)

q1 =  generate_input_vector(1000)         
tridiag_matrix = lanczos_iteration(1000, 1000, a, q1)

begin_time = timeit.default_timer()
eigenvalues, eigenvectors = np.linalg.eig(tridiag_matrix)
print(timeit.default_timer() - begin_time, "normal method")
eigenvalues = np.sort(eigenvalues)

begin_time = timeit.default_timer()
d = tridiag_matrix.diagonal()
e = np.empty(1000)
e[0:999] = tridiag_matrix.diagonal(-1)
e[-1] = 0.0 

z = np.zeros((len(d), len(d)), dtype=np.float64)  # Workspace for eigenvectors
ifst = 1  # Index of the first eigenvalue to be computed
ilst = len(d)  # Index of the last eigenvalue to be computed
il = 1  # Index of the first eigenvalue in the desired range
iu = len(d)  # Index of the last eigenvalue in the desired range
order = 'i'  # Compute eigenvalues and eigenvectors

# Call dstemr
geh, meh, guh, why = dstemr(d, e, len(d), ifst, ilst, il, iu)
meh = np.sort(meh)
# geh 1= number of eigenvalues/eigenvectors
# meh 2= eigenvalues
# guh 3= eigenvectors
# why 4=info, 22 which means something happened in DLARRV...
# BUT IT WORKING
print(why)

# evalues, evectors = dstemr(d, e)
print(timeit.default_timer() - begin_time, "dstemr method")
 
set_diff = meh - eigenvalues
print(np.around(set_diff, decimals=5))




# print("\nMinimum of scipy tridiag function: ", min(tridiag_times))
# print("\nAverage of scipy tridiag function: ", mean(tridiag_times))
# print("\nMinimum of general function: ", min(normal_times))
# print("\nAverage of general function: ", mean(normal_times))
