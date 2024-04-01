import pandas as pd
import matplotlib.pyplot as plt

# 1
df = pd.read_csv('data.csv', delimiter=',')

# 2
sal = df['MonthlyIncome']
print(f'{sal.describe()}')

# 3
age = df['age']
print(f'\n{df.head()}\n{age.tail(10)}')

# 5
new_debt = df[df['DebtRatio'] < 1]['DebtRatio'] * df['MonthlyIncome']
print(f'\nTask5\nNew debt:\n{new_debt}')

# 6
df.rename(columns={'DebtRatio': 'Debt'}, inplace=True)

# 7
salary = round(df['MonthlyIncome'].mean(), 2)
df['MonthlyIncome'] = df['MonthlyIncome'].fillna(salary)
print(salary)
print(df['MonthlyIncome'])

# 8
noreturn1 = df.groupby('NumberOfDependents')['SeriousDlqin2yrs'].mean()
noreturn2 = df.groupby('NumberRealEstateLoansOrLines')['SeriousDlqin2yrs'].mean()
print(f'\nTask 8:\nDependents: {noreturn1}\nLoans: {noreturn2}')

# 9a
'''
debt_f = df[df['SeriousDlqin2yrs'] == 0]
debt_t = df[df['SeriousDlqin2yrs'] == 1]

plt.scatter(debt_f['age'], debt_f['Debt'])
plt.scatter(debt_t['age'], debt_t['Debt'], color='red')
plt.title('Scatter')
plt.xlabel('Age')
plt.ylabel('Debt')
plt.show()
'''

# 9b
'''
df['MonthlyIncome'] = df['MonthlyIncome'].clip(upper=25000)
n_debt = df[df['SeriousDlqin2yrs'] == 0]['MonthlyIncome']
y_debt = df[df['SeriousDlqin2yrs'] == 1]['MonthlyIncome']

plt.hist(n_debt, bins=30, density=True, label='No Debt')
plt.hist(y_debt, bins=30, density=True, color='red', alpha=0.7, label='Debt')
plt.xlim(0, 25000)


plt.xlabel('Monthly Income')
plt.ylabel('Density')
plt.legend()
plt.show()
'''


# 9c

df['MonthlyIncome'] = df['MonthlyIncome'].clip(upper=25000)
pd.plotting.scatter_matrix(df[['age', 'MonthlyIncome', 'NumberOfDependents']], diagonal='kde')
plt.show()

