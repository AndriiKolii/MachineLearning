import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('student_scores.csv')

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

x_pred_train = regressor.predict(x_train)
y_pred_test = regressor.predict(x_test)

print(f'R^2 Train: {r2_score(y_train, x_pred_train)}')
print(f'R^2 Test: {r2_score(y_test, y_pred_test)}')

print(f'MSE Train: {mean_squared_error(y_train, x_pred_train)}')
print(f'MSE Test: {mean_squared_error(y_test, y_pred_test)}')


plt.subplot(2, 1, 1)
plt.scatter(x_train, y_train, color='green')
plt.plot(x_train, x_pred_train, color='blue')
plt.title('Train')
plt.xlabel('Hours')
plt.ylabel('Marks')

plt.subplot(2, 1, 2)
plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, y_pred_test, color='red')
plt.title('Test')
plt.xlabel('Hours')
plt.ylabel('Marks')
plt.tight_layout()
plt.show()
