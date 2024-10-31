import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

def train_decision_tree(csv):
    df = pd.read_csv(csv)
    X = df.iloc[:,[0,1,2,3,4,5]] 
    y = df.iloc[:, 6]   # Target (house price)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Decision Tree Regressor
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict house prices on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")

    
    corelation = df.iloc[:,0].corr(df.iloc[:,6])
    print(corelation)
    return model

def predict_value(value, model):
    value = np.array(value).reshape(1, -1)  # Reshape the input
    pred = model.predict(value)
    print(f"Predicted House Price: {pred[0]}")

# Train the model and make predictions
model = train_decision_tree('re.csv')
value = [2013.5,8.5, 104.81, 5, 24.97, 121.54]  # Example input
predict_value(value, model)

