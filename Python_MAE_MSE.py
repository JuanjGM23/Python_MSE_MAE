import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Generate regression data
X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
np.random.seed(0)
y_test = y_test + np.random.normal(size=len(y_test)) * 0.1

# Plot train / test data
print(X_train.shape, X_test.shape , y_train.shape, y_test.shape)
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, label='Training Data', color='blue', alpha=0.7)
plt.scatter(X_test, y_test, label='Testing Data', color='red', alpha=0.7)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression Data: Training vs. Testing')
plt.legend()
plt.pause(0.1)  # Pausa para mostrar la gráfica en una ventana emergente

# Compute MAE and MSE
X, y = make_regression(n_samples=1000, n_features=1, noise=10, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
np.random.seed(0)
y_test = y_test + np.random.normal(size=len(y_test)) * 0.1

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)

# Plot train / test data with regression line
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, label='Training Data', color='blue', alpha=0.7)
plt.scatter(X_test, y_test, label='Testing Data', color='red', alpha=0.7)
plt.plot(X, regressor.predict(X), color='green', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regression Data: Training vs. Testing')
plt.legend()
plt.pause(0.1)  # Pausa para mostrar la grafica en una ventana
# Sirve para esperar que el usuario cierre las ventanas emergentes
plt.show(block=True)
