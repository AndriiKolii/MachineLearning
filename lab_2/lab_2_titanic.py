import pandas as pd
import numpy as np


df = pd.read_csv('titanic.csv')
male = (df['sex'] == 'male').sum()
female = (df['sex'] == 'female').sum()
print(f'Male: {male}, Female: {female}')

survivors = (df['survived'] == 1).sum() / len(df) * 100
print(f'{round(survivors, 2)} survived')

first = (df['pclass'] == 1).sum() / len(df) * 100
print(f'1-st class passengers: {round(first, 2)}')

age = df['age'].dropna()
average = np.average(age)
median = np.median(age)
print(f'Average: {round(average, 2)}, Median: {median}')
