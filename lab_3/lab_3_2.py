import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('petrol_consumption.xls')

x = df.drop('Petrol_Consumption', axis=1)
y = df['Petrol_Consumption']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=1)

model = LinearRegression()
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print(f'R^2 Train: {r2_score(y_train, y_train_pred)}')
print(f'R^2 Test: {r2_score(y_test, y_test_pred)}')

print(f'MSE Train: {mean_squared_error(y_train, y_train_pred)}')
print(f'MSE Test: {mean_squared_error(y_test, y_test_pred)}')


plt.subplot(2, 1, 1)
plt.scatter(y_train, y_train_pred, color='green')
plt.plot(y_train, y_train, color='blue')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Train')

plt.subplot(2, 1, 2)
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot(y_test, y_test, color='red')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Test')
plt.tight_layout()
plt.show()
