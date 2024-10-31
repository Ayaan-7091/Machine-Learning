import pandas as pd

df = pd.read_csv('HousingData.csv')

selected_features = ["RM","TAX","DIS","RAD","AGE"]

df[selected_features].isnull().sum()

df = df.dropna(subset=selected_features)
# print(df)

# Now, let's create a data frame with selected values
X = df[["RM","DIS"]]
y = df["MEDV"] #target value

# spitting the data into training and testing sets 
# print(x)
# print(y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
#Fit the model to the training data
model.fit(X_train,y_train)

# custom value prediction
custom_value = [[6.58,4.09]]


#Make prediction on the test set
y_pred = model.predict(X_test)
y_custom_pred = model.predict(custom_value)

# custom value prediction
# y_pred = model.predict(custom_value)

# evaluating the model's performance
from sklearn.metrics import mean_absolute_error,mean_squared_error

mse = mean_squared_error(y_test,y_pred)
mae = mean_absolute_error(y_test,y_pred)

# print(f"Predicted value : {y_pred}")
print(f"mean squared error : {mse}")
print(f"mean absolute error : {mae}")
print(f"Predicted Value : {y_custom_pred}")

import matplotlib.pyplot as plt

# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual MEDV Values")
# plt.ylabel("Predicted MEDV Values")
# plt.title("Actual vs Predicted MEDV")
# plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # Diagonal line
# plt.show()


# checking the Linear Relationship between the feature and target 
plt.scatter(df["AGE"],df["MEDV"]) 
plt.xlabel("Rooms")
plt.ylabel("House Price")
plt.title("Linearity Checker")
plt.show()

coefficient = df["AGE"].corr(df["MEDV"])
print(coefficient)