import numpy as np

x = np.random.randint(-15, 15, size=(4, 5))
print(x)

rows, columns = x.shape

n = np.size(x)
for i in range(0, rows):
    for j in range(0, columns):
        if x[i][j] < 0:
            x[i][j] = -1
        elif x[i][j] > 0:
            x[i][j] = 1
print(f'New Array:\n{x}')
