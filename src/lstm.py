import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from matplotlib.pylab import rcParams
from sklearn.preprocessing import MinMaxScaler

from sklearn import preprocessing;
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import linear_model;

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

####read data

df = pd.read_csv(sys.argv[1])

####visualization

#setting figure size
rcParams['figure.figsize'] = 20,10

#for normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#plot
#plt.plot(df['Close'], label='Close Price history')
#plt.show()

####lstm
# 导入 keras 等相关包
# 选取 date 和 close 两列
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

# setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

# 分成 train and test
dataset = new_data.values

train = dataset[:-60,:]
test = dataset[-60:,:]

# 构造 x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# 建立 LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

inputs = new_data[len(new_data) - len(test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

with open(sys.argv[2],'w') as f:
    for i in range(len(train)-len(test)):
        f.write(str(train[i][0])+' '+str(-1)+'\n')
    for i in range(len(train)-len(test),len(train)):
        f.write(str(train[i][0])+' '+str(test[i-(len(train)-len(test))][0])+'\n')

'''
#for plotting
train = new_data[:-60]
test = new_data[-60:]
test['Predictions'] = closing_price
plt.figure(figsize=(12,4))
plt.plot(train['Close'],color='b')
plt.plot(test[['Close','Predictions']],color='r')
plt.show()
'''
