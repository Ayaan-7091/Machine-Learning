import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression


def train_multiple_linear_regressions(csv):
    df = pd.read_csv(csv)
    # print(df.isnull().sum())
    x = df.iloc[:,[0,1,2,3,4,5]]
    y = df.iloc[:,6]

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=42)

    model = LinearRegression()
    
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    
    print(f"mean squared error : {mse}")
    print(f"mean absolute error : {mae}")

    return model

def predict_value(value,model):
    value = np.array(value).reshape(1,-1)
    v_pred = model.predict(value)
    print(f"Predicted value : {v_pred}")
    

model = train_multiple_linear_regressions('re.csv')
value = [2018,5, 20, 8, 24.98298, 121.54024]  # Example input
predict_value(value,model)
    

