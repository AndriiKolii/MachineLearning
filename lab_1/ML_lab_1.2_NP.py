import numpy as np

x = np.arange(12.5, 5, -0.5)
y = x - np.average(x)

print(np.average(x))
print(f'Array lenght: {np.size(x)}\nNew Array: {np.sort(y)}')
