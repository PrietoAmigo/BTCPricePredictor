import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import datetime

# CREATING THE DATAFRAME TO STORE DATA
df = yf.download('BTC-USD', start='2013-01-01', end=datetime.date.today(), progress=False)

print(df.columns.tolist())

# df.index = df.index.strftime('%d/%m/%Y')



# PV = plt.plot(df["Close"], label='Close Price history')
# plt.show()



df = df.sort_index(ascending=True,axis=0)
data = pd.DataFrame(index=range(0, len(df)),columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
for i in range(0, len(data)):
    data["Close"][i] = df["Close"][i]
    data["High"][i] = df["High"][i]
    data["Low"][i] = df["Low"][i]
    data["Open"][i] = df["Open"][i]
    data["Adj Close"][i] = df["Adj Close"][i]
    data["Volume"][i] = df["Volume"][i]

# print(data.head())
# print(data.tail())

scaler = MinMaxScaler(feature_range=(0, 1))

# data.index = data.Date
# data.drop("Date", axis=1, inplace=True)

final_data = data.values

train_data = final_data[0:200, :]
valid_data = final_data[200:, :]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_data)
x_train_data, y_train_data=[], []

for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i, 0])
    y_train_data.append(scaled_data[i, 0])


lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(np.shape(x_train_data)[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
model_data = data[len(data)-len(valid_data)-60:].values
model_data = model_data.reshape(-1, 1)
model_data = scaler.transform(model_data)

lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)
X_test = []
for i in range(60, model_data.shape[0]):
    X_test.append(model_data[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))