import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Data
df = pd.read_csv('re.csv')

# Extract Numeric Data
df['N_RAM'] = df['Ram'].str.extract('(\d+)').astype(int)
df['N_BATTERY'] = df['Battery'].str.extract('(\d+)').astype(int)
df['N_DISPLAY'] = df['Display'].str.extract('([0-9]*\.?[0-9]+)').astype(float)
df.dropna(subset=['Android_version', 'Inbuilt_memory'], inplace=True)
df['N_VERSION'] = df['Android_version'].str.extract('(\d+)').astype(float)


# Convert Inbuilt Memory to GB scale
extracted_memory_info = df['Inbuilt_memory'].str.extract('(\d+)\s*(GB|TB)')
extracted_memory_info[0] = extracted_memory_info[0].astype(int)
df['N_MEMORY'] = extracted_memory_info.apply(lambda x: x[0]*1024 if x[1] == 'TB' else x[0], axis=1)

# Fast Charging, fill missing with 5
df['N_CHARGING'] = df['fast_charging'].str.extract('(\d+)').astype(float).fillna(5)

# Camera Processing
df[['temp_REAR_MP','temp_FRONT_MP']] = df['Camera'].str.split('&', expand=True)
df['N_FRONT_MP'] = df['temp_FRONT_MP'].str.extract('(\d+)').astype(float).fillna(0)
df['REAR_MP_VALUES'] = df['temp_REAR_MP'].str.findall('(\d+)\s*MP').apply(lambda x: list(map(int, x)))
df = df[df['REAR_MP_VALUES'].str.len() > 0]
df['N_REAR_MP'] = df['REAR_MP_VALUES'].apply(max)

# Target Variable and Log Transformation
df['Y'] = df['Price'].str.replace(',', '').astype(int)
df['Y'] = np.log(df['Y'])  # Log transformation

# Rank Normalization of Features
for column in ['N_RAM', 'N_BATTERY', 'N_DISPLAY', 'N_MEMORY', 'N_CHARGING', 'N_FRONT_MP', 'N_REAR_MP']:
    df[column] = rankdata(df[column]) / len(df[column]) * 100

# Prepare Data for Model
X = df[['Rating', 'Spec_score', 'N_RAM', 'N_BATTERY', 'N_DISPLAY', 'N_VERSION', 'N_MEMORY', 'N_CHARGING', 'N_FRONT_MP', 'N_REAR_MP']]
y = df['Y']

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and Evaluate Model
model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

# Mean Squared Error
mse = mean_squared_error(Y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Inverse Log Transformation to Compare Actual vs Predicted Prices
pred_prices = np.exp(y_pred)
actual_prices = np.exp(Y_test)

# Show Results
result_df = pd.DataFrame({'Actual Price': actual_prices, 'Predicted Price': pred_prices})
print(result_df.head(10))
