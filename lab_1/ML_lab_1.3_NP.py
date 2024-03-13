import numpy as np

x = np.random.randint(1, 10, size=(1, 20))
print(f'Old Array:\n{x}')

x = np.reshape(x, [4, 5])
print(f'New Array:\n{x+10}')
