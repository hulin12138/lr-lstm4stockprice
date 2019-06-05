import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read data

df = pd.read_csv('acglo.csv')
df.head()
df.describe()

#visualization

#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close Price'], label='Close Price history')
