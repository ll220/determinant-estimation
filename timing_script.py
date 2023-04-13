import numpy as np
import time
from statistics import mean
from matplotlib.pyplot import plot 

ITERATIONS = 30

def generate_matrix(size):
    triangle_matrix = generate_lower_triangle_matrix(size)
    transpose = triangle_matrix.transpose()
    
    pos_semi_def_matrix = np.matmul(triangle_matrix, transpose)

    # print("Random Triangle matrix: \n", triangle_matrix)
    # print("\nTranspose: \n", transpose)
    # print("\n\n Random Positive Semi-Definite matrix size ", size, ": \n", pos_semi_def_matrix)
    return pos_semi_def_matrix

def generate_lower_triangle_matrix(size):
    triangle_matrix = np.zeros(shape=(size, size), dtype=int)
    for x in range(size):
        for y in range(size - x):
            triangle_matrix[size - x - 1][y] = np.random.randint(-50, 50)
    return triangle_matrix

times = []
sizes = []

for x in range(1, 100, 2):
    iterations = []
    for y in range(30):
        matrix = generate_matrix(x)

        start_time = time.time()
        determinant = np.linalg.det(matrix)
        # print(x, " ", determinant, "\n")
        end_time = time.time()

        # print(end_time - start_time, "\n")
        iterations.append(end_time - start_time)
    
    sizes.append(x)
    times.append(mean(iterations))

# plot(sizes, times)
print(sizes)
print(times)

# matrix = generate_matrix(5)
# print(matrix)
# determinant = np.linalg.det(matrix)
# print(determinant)