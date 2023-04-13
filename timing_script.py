import numpy as np
import time
from statistics import mean
import matplotlib.pyplot as plt

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
    triangle_matrix = np.zeros(shape=(size, size))
    for x in range(size):
        for y in range(size - x):
            triangle_matrix[size - x - 1][y] = np.random.normal(loc=0.0, scale=1.0, size=None)

            if (size - x - 1) == y and triangle_matrix[size - x - 1][y] < 0.0:
                triangle_matrix[size - x - 1][y] = 1.0

    return triangle_matrix

times = []
sizes = []

for x in range(1, 100, 2):
    iterations = []
    for y in range(30):
        matrix = generate_matrix(x)

        start_time = time.time()
        determinant = np.linalg.det(matrix)
        # print(determinant, "\n")
        end_time = time.time()

        # print(end_time - start_time, "\n")
        iterations.append(end_time - start_time)
    
    sizes.append(x)
    times.append(mean(iterations))


plt.plot(sizes, times)
plt.title('Determinant Computation Times')
plt.xlabel('Sizes')
plt.ylabel('Time (sec)')
print(sizes)
print(times)
plt.show()

# matrix = generate_matrix(3)
# print(matrix)
# determinant = np.linalg.det(matrix)
# print(determinant)