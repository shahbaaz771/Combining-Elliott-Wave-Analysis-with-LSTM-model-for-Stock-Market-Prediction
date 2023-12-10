# %%
#import required libraries
import numpy as np
import matplotlib.pyplot as plt
import taew
import pandas_datareader.data as web
import yfinance as yfn
import pandas as pd

yfn.pdr_override()

# get the stock price of apple company 1 years
start_date = '2015-1-1'
end_date = '2016-1-1'

stock_data = web.get_data_yahoo('AAPL', start_date, end_date) #df= Data frame


print(type(stock_data))

stock_data = stock_data['Close']


# %%
#Cleaning the data
stock_data = stock_data.dropna()
print(stock_data.isnull().sum())
print(stock_data)

# %%
prices = np.array(stock_data.values, dtype=np.double).flatten(order='C')


# Identify the upward Elliott wave using the Alternative_ElliottWave_label_upward method from taew library
waves = taew.Alternative_ElliottWave_label_upward(prices)

# %%
waves

# %%
# Extract the x and z values from the identified waves
x_values = []
z_values = []

for wave in waves:
    x_values.extend(wave['x'])
    z_values.extend(wave['z'])


# %%
x_values

# %%
# Function to find the buying point and the selling point in the elliott wave points we found
def retracement(x_values,z_values):
    buyingPoint = []
    buyingIndex = []
    sellingPoint = []
    sellingIndex = []
    for i in range(0,int(len(x_values)/6)):
        point0 = x_values[(6*(i))] 
        point1 = x_values[(6*(i)) + 1] 
        point2 = x_values[(6*(i)) + 2]
        point3 = x_values[(6*(i)) + 3]
        point4 = x_values[(6*(i)) + 4]
        point5 = x_values[(6*(i)) + 5]

        index0 = z_values[(6*(i))] 
        index1 = z_values[(6*(i)) + 1] 
        index2 = z_values[(6*(i)) + 2]
        index3 = z_values[(6*(i)) + 3]
        index4 = z_values[(6*(i)) + 4]
        index5 = z_values[(6*(i)) + 5]
        # print(point0,point1,point2)
        # print(index0,index1,index2)
        wave1 = point1-point0
        wave2 = point1-point2 
        wave3 = point3-point2
        wave4 = point4-point3
        wave5 = point5-point4

        if wave2 <= wave1*0.618:
            buyingPoint.append(point2)
            buyingIndex.append(index2)
            
        if wave5 >= wave4*0.382:
            sellingPoint.append(point5)
            sellingIndex.append(index5)

        
        
        
    return buyingPoint,buyingIndex,sellingPoint,sellingIndex

# %%
# Function to randomly pick 6 points from the x_values to plot in the graph
from random import randint

def randomize_wave_plot(x_values,z_values):

    # Get the indices that are multiples of 6
    xv = []
    zv = []
    temp = randint(1,len(x_values)/6)

    for i in range(temp*6-6,temp*6):
        xv.append(x_values[i])
        zv.append(z_values[i])
    
    return xv,zv

# %%
#randomize elliott wave plot
xv,zv = randomize_wave_plot(x_values,z_values)

buy,buy_point,sell,sell_point = retracement(x_values,z_values)

# Visualization of the elliott wave analysis
plt.figure(figsize=(16,8))
fig, ax = plt.subplots()
ax.plot(prices, label='Stock Market Prices')
ax.scatter(z_values,x_values,color='red', label='Identified Elliot Waves point')
ax.scatter(buy_point,buy,color='green', label='Buying Point')
ax.scatter(sell_point,sell,color='orange', label='Buying Point')
ax.plot(zv,xv,c='black',label="Elliott Wave")
ax.legend()

# Set labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.set_title('Stock Market Trend with Elliott Waves')

plt.show()

# %%
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers import LSTM

# %%
x_values

# %%
last_12_values = x_values[-6:]
print(last_12_values)

# %%
del x_values[-6:]

# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Convert the list to a pandas Series
x_values_series = pd.Series(x_values)


# %%
# Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaler_data = scaler.fit_transform(x_values_series.values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaler_data)):
    x_train.append(scaler_data[x-prediction_days:x, 0])
    y_train.append(scaler_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# %%
scaler_data

# %%
# Build a model #
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  #Prediction of the nxt close price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# %%
last_12_values = pd.Series(last_12_values)

# %%
# Test the model accuracy on existing data #
# load test data

actual_prices = last_12_values.values

total_dataset = pd.concat((x_values_series, last_12_values), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(last_12_values) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# %%
last_12_values

# %%
model_inputs

# %%
# Make Predictions on test data #
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

# %%
prediction_prices

# %%
last_12_values

# %%
# plot the test predictions #
plt.plot(actual_prices, color="black", label=f"Actal AAPL Price")
plt.plot(prediction_prices, color="green", label=f"Predicted AAPL Price")
plt.title("AAPL Stock Market Price")
plt.xlabel("Time")
plt.ylabel("AAPL Share Price")
plt.legend()
plt.show()

# %%
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_prices, prediction_prices))

# Calculate MAE
mae = mean_absolute_error(actual_prices, prediction_prices)

# Calculate MAPE
mape = (mae / np.mean(actual_prices)) * 100

print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape)

# %%
# predict the next day #
real_data = [model_inputs[len(model_inputs)+1-prediction_days:len(model_inputs+1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

print(scaler.inverse_transform(real_data[-1]))
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"nxt prediction : {prediction}")


