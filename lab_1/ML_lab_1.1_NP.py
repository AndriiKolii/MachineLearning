import numpy as np

x = np.array([-1, 3])
y = np.array([4, 6])
mult = 1.

a = np.concatenate((x, y))
maximum = np.max(a)
minimum = np.min(a)
summ = np.sum(a)

for i in range(0, np.size(a)):
    mult *= a[i]

print(f'Sum: {x + y}\nDif: {x - y}\nMult: {x * y}\nDiv: {x / y}\n')
print(f'Concatenate: {a}\nMax: {maximum}\nMin: {minimum}\nSum: {summ}\nMult: {mult}')
