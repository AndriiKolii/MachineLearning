import numpy as np

x = np.array([[1, 1, 2, 3],
              [3, -1, -2, -2],
              [2, -3, -1, -1],
              [1, 2, 3, -1]])
y = np.array([[1],
              [-4],
              [-6],
              [-4]])

print(np.linalg.solve(x, y))
