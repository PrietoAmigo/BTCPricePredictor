import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime
from yahoofinancials import YahooFinancials


#CREATING THE DATAFRAME TO STORE DATA
df = yf.download('BTC-USD',
   start='2013-01-01',
   end=datetime.date.today(),
   progress=False)
df.head()

print(df.columns.tolist())


series = df['Close'].values.reshape(-1, 1)


scaler = StandardScaler()
scaler.fit(series[:len(series) // 2])
series = scaler.transform(series).flatten()


T = 10
D = 1
X = []
Y = []



for t in range(len(series) - T):
   x = series[t:t + T]
   X = np.append(X, x)
   y = series[t + T]
   Y = np.append(Y, y)
   X = np.array(X).reshape(-1, T)
   Y = np.array(Y)
   N = len(X)
   print("X.shape", X.shape, "Y.shape", Y.shape)


class BaselineModel:
 def predict(self, X):
   return X[:,-1] # return the last value for each input sequence


Xtrain, Ytrain = X[:-N//2], Y[:-N//2]
Xtest, Ytest = X[-N//2:], Y[-N//2:]


model = BaselineModel()
Ptrain = model.predict(Xtrain)
Ptest = model.predict(Xtest)


Ytrain2 = scaler.inverse_transform(Ytrain.reshape(-1, 1)).flatten()
Ytest2 = scaler.inverse_transform(Ytest.reshape(-1, 1)).flatten()
Ptrain2 = scaler.inverse_transform(Ptrain.reshape(-1, 1)).flatten()
Ptest2 = scaler.inverse_transform(Ptest.reshape(-1, 1)).flatten()



# right forecast
forecast = []
input_ = Xtest[0]
while len(forecast) < len(Ytest):
 f = model.predict(input_.reshape(1, T))[0]
 forecast.append(f)
 # make a new input with the latest forecast
 input_ = np.roll(input_, -1)
 input_[-1] = f
plt.plot(Ytest, label='target')
plt.plot(forecast, label='prediction')
plt.legend()
plt.title("Right forecast")
plt.show()