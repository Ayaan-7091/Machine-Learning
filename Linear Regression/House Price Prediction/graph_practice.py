import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('HousingData.csv')

# Select the features and target
X = df[['RM']]  # Using 'RM' (number of rooms) as feature
y = df['MEDV']  # Target: House price

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Plot the actual data points
plt.scatter(X_test, y_test, color='blue', label='Actual Data')

# Plot the regression line
plt.plot(X_test, y_pred, color='red', label='Regression Line')

# Adding labels and title
plt.xlabel('Number of Rooms (RM)')
plt.ylabel('House Price (MEDV)')
plt.title('Linear Regression: House Price vs. Number of Rooms')
plt.legend()

# Show the plot
plt.show()
