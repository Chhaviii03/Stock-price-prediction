import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import warnings
import zipfile
import os

warnings.filterwarnings("ignore")

zip_file_path = 'all_stocks_5yr.csv.zip'
csv_file_name = 'all_stocks_5yr.csv'

if not os.path.exists(csv_file_name):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall()

# Loading dataset
data = pd.read_csv(csv_file_name, delimiter=',', on_bad_lines='skip') 
print(data.shape)  # Printing the shape of dataset
print(data.sample(7))  # Print a random sample of 7 rows from the dataset

# Show dataset information
data.info()

# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])
data.info()  # Check the data info after conversion

# Plot date vs open and close prices for specific companies
companies = ['AAPL', 'AMD', 'FB', 'GOOGL', 'AMZN', 'NVDA', 'EBAY', 'CSCO', 'IBM']

plt.figure(figsize=(15, 8)) 
for index, company in enumerate(companies, 1): 
    plt.subplot(3, 3, index) 
    c = data[data['Name'] == company] 
    plt.plot(c['date'], c['close'], c="r", label="close", marker="+") 
    plt.plot(c['date'], c['open'], c="g", label="open", marker="^") 
    plt.title(company) 
    plt.legend() 
    plt.tight_layout()

plt.show()

# Plot date vs volume for the same companies
plt.figure(figsize=(15, 8))
for index, company in enumerate(companies, 1):
    plt.subplot(3, 3, index)
    c = data[data['Name'] == company]
    plt.plot(c['date'], c['volume'], c='purple', marker='*')
    plt.title(f"{company} Volume")
    plt.tight_layout()

plt.show()

# Filter Apple stock data for a specific date range
apple = data[data['Name'] == 'AAPL']
prediction_range = apple.loc[(apple['date'] > datetime(2013,1,1)) & (apple['date'] < datetime(2018,1,1))]

# Plot Apple stock closing price
plt.plot(apple['date'], apple['close'])
plt.xlabel("Date")
plt.ylabel("Close")
plt.title("Apple Stock Prices")
plt.show()

# Prepare data for training the LSTM model
close_data = apple.filter(['close'])
dataset = close_data.values
training = int(np.ceil(len(dataset) * .95))
print(training)

# Scale the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Split the data into training and testing sets
train_data = scaled_data[0:int(training), :]
x_train = []
y_train = []

# Prepare feature and label sets
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Define the LSTM model
model = keras.models.Sequential()
model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=64))
model.add(keras.layers.Dense(32))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(x_train, y_train, epochs=10)

# Prepare the test dataset
test_data = scaled_data[training - 60:, :]
x_test = []
y_test = dataset[training:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict the testing data
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Calculate MSE and RMSE
mse = np.mean(((predictions - y_test) ** 2))
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))

# Compare predicted and actual data
train = apple[:training]
test = apple[training:]
test['Predictions'] = predictions

# Plot the results
plt.figure(figsize=(10, 8))
plt.plot(train['date'], train['close'])
plt.plot(test['date'], test[['close', 'Predictions']])
plt.title('Apple Stock Close Price')
plt.xlabel('Date')
plt.ylabel("Close")
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()

# End of script
