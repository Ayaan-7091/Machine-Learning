import pandas as pd
import numpy as np
df = pd.read_csv('re.csv')

# print(df.head())

df['N_RAM'] = df['Ram'].str.extract('(\d+)').astype(int)
df['N_BATTERY'] = df['Battery'].str.extract('(\d+)').astype(int)
df['N_DISPLAY'] = df['Display'].str.extract('([0-9]*\.?[0-9]+)').astype(float)

df.dropna(subset=['Android_version'], inplace=True)
df['N_VERSION'] = df['Android_version'].str.extract('(\d+)').astype(float)

#Inbuilt Memory
df.dropna(subset=['Inbuilt_memory'],inplace=True)
extracted_memory_info = df['Inbuilt_memory'].str.extract('(\d+)\s*(GB|TB)')
#converting memory value to int
extracted_memory_info[0] = extracted_memory_info[0].astype(int)

def memory_converter(row):
    value = row[0]
    unit = row[1]

    return value*1024 if unit == 'TB' else value

df['N_MEMORY'] = extracted_memory_info.apply(memory_converter,axis=1)

df.dropna(subset=['fast_charging'],inplace=True)
df['N_CHARGING'] = df['fast_charging'].str.extract('(\d+)').astype(float)

# Camera 
df.dropna(subset=['Camera'],inplace=True)
df[['temp_REAR_MP','temp_FRONT_MP']] = df['Camera'].str.split('&',expand=True)


df['N_FRONT_MP'] = df['temp_FRONT_MP'].str.extract('(\d+)').astype(float)
df['N_FRONT_MP'].fillna(0, inplace=True)

df['REAR_MP_VALUES'] = df['temp_REAR_MP'].str.findall('(\d+)\s*MP').apply(lambda x:list(map(int,x)))
df = df[df['REAR_MP_VALUES'].str.len() > 0]
df['N_REAR_MP'] = df['REAR_MP_VALUES'].apply(max)

df.dropna(subset=['Rating'],inplace=True)
df.dropna(subset=['Rating','Spec_score','N_RAM','N_BATTERY','N_DISPLAY','N_VERSION','N_MEMORY','N_CHARGING','N_FRONT_MP','N_REAR_MP'],inplace=True)


df['Y'] = df['Price'].str.replace(',','').astype(int)

X = df[['Rating','Spec_score','N_RAM','N_BATTERY','N_DISPLAY','N_VERSION','N_MEMORY','N_CHARGING','N_FRONT_MP','N_REAR_MP']]
y = np.log(df['Y'])

print(X)
print(y)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(Y_test,y_pred)

Y_PREDICTED = np.exp(y_pred)
Y_ACTUAL = np.exp(Y_test)
print(f'mse : {mse}')

result_df = pd.DataFrame({'Predicted Prices':Y_PREDICTED,'Actual Prices':Y_ACTUAL})
print(result_df.head(25))