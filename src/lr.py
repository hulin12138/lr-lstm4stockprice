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


####read data

df = pd.read_csv(sys.argv[1])



#for normalizing data
scaler = MinMaxScaler(feature_range=(0, 1))

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']


#####liner regretion
def prepare_data(df,forecast_col,forecast_out,test_size):
    label = df[forecast_col].shift(-forecast_out);      # 建立 label，是 forecast_col 这一列的向右错位 forecast_out=5 个位置，多出的是 na
    X = np.array(df[[forecast_col]]);                   # X 为 是 forecast_col 这一列
    X = preprocessing.scale(X)                          # processing X
    X_lately = X[-forecast_out:]                        # X_lately 是 X 的最后 forecast_out 个数，用来预测未来的数据
    X = X[:-forecast_out]                               # X 去掉最后 forecast_out 几个数
    label.dropna(inplace=True);                         # 去掉 na values
    y = np.array(label)                                
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size) 
    #X_train = X[:-60]
    #X_test = X[-60:]
    #Y_train = y[:-60]
    #Y_test = y[-60:]	
    response = [X_train,X_test , Y_train, Y_test , X_lately];
    return response;

forecast_col = 'Close'                            # 选择 close 这一列
forecast_out = 5                                        # 要预测未来几个时间步 
test_size = 0.2;                                        # test set 的大小
print("start preparing data for the model")
X_train, X_test, Y_train, Y_test , X_lately =prepare_data(df,forecast_col,forecast_out,test_size)
'''
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
print(X_lately)
'''
print("model data prepared")
model = linear_model.LinearRegression();              
print("training model... ... ")
model.fit(X_train,Y_train);
print("model training done")
score = model.score(X_test,Y_test);
print(score)        

y_test_predict = model.predict(X_test)

print(y_test_predict.shape)
print(Y_test.shape)
rcParams['figure.figsize'] = 20,10
plt.figure(figsize=(12,4))
plt.plot(y_test_predict,color='r')
plt.plot(Y_test,color='b')
#for i in range(len(Y_test)):
#    print(y_test_predict[i],Y_test[i])
plt.show()

forecast= model.predict(X_lately)
print(forecast)
