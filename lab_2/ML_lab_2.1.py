import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
y = 1 / (x * np.sin(5 * x))

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Function')
plt.plot(x, y)
plt.show()
