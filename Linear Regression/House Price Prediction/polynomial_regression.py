from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Define the feature and target
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('re.csv')

X = df[['X4 number of convenience stores']]  # Replace with the correct column name
y = df['Y house price of unit area']  # Replace with the correct target column name

# Transform the feature to include polynomial terms
poly = PolynomialFeatures(degree=2)  # You can change the degree as needed
X_poly = poly.fit_transform(X)

# Fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Predict and evaluate
y_pred = model.predict(X_poly)
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")

# Plotting
plt.scatter(X, y, color='blue', label='Data Points')
plt.scatter(X, y_pred, color='red', label='Polynomial Fit')
plt.xlabel('Number of Convenience Stores')
plt.ylabel('House Price')
plt.legend()
plt.show()
