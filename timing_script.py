import numpy as np
import time
from statistics import mean
import matplotlib.pyplot as plt

ITERATIONS = 30

RANDOM_MEAN = 0.0
RANDOM_ST = 5.0

MAXIMUM = 500
STEP_SIZE = 20
# tend to start hitting the overflow errors with standard deviation 5.0 and mean 0.0 at around max size 190

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
            triangle_matrix[size - x - 1][y] = np.random.normal(loc=RANDOM_MEAN, scale=RANDOM_ST, size=None)

            if (size - x - 1) == y and triangle_matrix[size - x - 1][y] < 0.0:
                triangle_matrix[size - x - 1][y] = 1.0

    return triangle_matrix

times = []
sizes = []

f = open("demofile2.txt", "a")
f.write("Determinants: \n")


for x in range(1, MAXIMUM, STEP_SIZE):
    iterations = []
    for y in range(30):
        matrix = generate_matrix(x)

        start_time = time.time()
        determinant = np.linalg.det(matrix)
        end_time = time.time()
        f.write(str(determinant.item()))
        f.write("\n")

        # print(end_time - start_time, "\n")
        iterations.append(end_time - start_time)
    
    sizes.append(x)
    times.append(mean(iterations))

plot_title = "Determinant Computation Times Standard dev: " + str(RANDOM_ST)

plt.plot(sizes, times)
plt.title(plot_title)
plt.xlabel('Sizes')
plt.ylabel('Time (sec)')
# print(sizes)
# print(times)
plt.show()

f.close()

# matrix = generate_matrix(3)
# print(matrix)
# determinant = np.linalg.det(matrix)
# print(determinant)