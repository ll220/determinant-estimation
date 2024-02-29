import numpy as np


A = np.matrix([[3, 2], [2, 6]])
b = np.matrix([[2], [-8]])

r = b
p = r

alpha = None
x = np.matrix([[0], [0]])

ITERS = 5

for n in range(1, ITERS):
    alpha = ((r.T @ r) / (p.T @ A @ p)).item()
    # print("alpha:", alpha)
    x = x + alpha * p
    # print("x: ", x)
    r_new = r - alpha * (A @ p)
    # print("r_new: ", r_new)
    beta = ((r_new.T @ r_new) / (r.T @ r)).item()
    # print("beta: ", beta)
    p = r_new + beta * p
    # print("p: ", p)
    r = r_new

# print(x)