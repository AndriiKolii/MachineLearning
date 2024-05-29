import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def function(x):
    return x ** 3 - 10 * x ** 2 + x


x_values = np.arange(0, 10, 0.1)
y_values = function(x_values)

x_train = x_values[::2]
y_train = y_values[::2]

X_train = np.column_stack([x_train ** i for i in range(14)])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

X_values = np.column_stack([x_values ** i for i in range(14)])
X_values_scaled = scaler.transform(X_values)
predictions = lr.predict(X_values_scaled)

plt.plot(x_values, predictions)
plt.scatter(x_values, predictions, color='red', marker='o')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Prediction')
plt.tight_layout()
plt.show()

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
ridge_predictions = ridge.predict(X_values_scaled)

lasso = Lasso(alpha=1.0, max_iter=10000)
lasso.fit(X_train_scaled, y_train)
lasso_predictions = lasso.predict(X_values_scaled)

ridge_r2 = r2_score(y_values, ridge_predictions)
lasso_r2 = r2_score(y_values, lasso_predictions)
r2 = r2_score(y_values, predictions)
print(f'r2: {r2}\nr2 ridge: {ridge_r2}\nr2 lasso: {lasso_r2}')
