from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
import pandas as pd


digits = load_digits()
scaled = scale(digits.data)

default_values = pd.DataFrame(digits.data, columns=[f'val_{i}' for i in range(digits.data.shape[1])])
scaled_values = pd.DataFrame(scaled, columns=[f'val_{i}' for i in range(digits.data.shape[1])])

print(f'Default: {default_values.head()}')
print(f'Scaled: {scaled_values.head()}')
