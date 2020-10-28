#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas_datareader as web
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
warnings.filterwarnings('ignore')


# In[18]:


Data = web.DataReader("MSFT",data_source='yahoo',start = '2010-01-10',end = '2020-10-25')
Data


# In[19]:


#plotting
import matplotlib.pyplot as plt
plt.figure(figsize = (16,8))
plt.plot(Data['Close'], label = 'CLOSE PRICE HISTORY')


# In[20]:


new_data = pd.DataFrame(index = range(0,len(Data)),columns = ['Close'])
for i in range(0,len(Data)):
    new_data['Close'][i] = Data['Close'][i]
new_data.index = Data.index
new_data


# In[21]:


#setting test and train data 
train = new_data[:2500]
test = new_data[2500:]
print(train.shape)
print(test.shape)


# In[22]:


#now we will work on moving average and then 
#we approach root mean square error
pred = []
for i in range(0,test.shape[0]):
    a = train['Close'][len(train)-len(test)+i:].sum() + sum(pred)
    b = a/len(test)
    pred.append(b)

rms = np.sqrt(np.mean(np.power((np.array(test['Close'])-pred),2)))
print('rms: ', rms)


# In[30]:


#RMSE does not give complete information, so we visualize this model
test['Predictions'] = pred
plt.figure(figsize=(16,8))
plt.plot(train['Close'])
plt.plot(test[['Close','Predictions']])
plt.show()


# In[24]:


#Inference
#We can see that results are not very promising
#though the prediction is in the range of observed data
#and the prediction curve show some increase in stock price and then a slight decrese
#but it is not what we want


# In[25]:


#so we will try it with different ML techniques
#Auto-ARIMA and LSTM


# In[26]:


#creating new dataset so that it don't affect original dataset
df = pd.DataFrame(index = range(0,len(Data)),columns = ['Close'])
df.index =Data.index

for i in range(len(Data)):
    df['Close'][i] = Data['Close'][i]
df = df.reset_index()
df


# In[28]:


#let's try with Auto ARIMA
from pmdarima import auto_arima
a_data = df

train = a_data[:2500]
test = a_data[2500:]

training = train['Close']
testing = test['Close']


# In[29]:


model = auto_arima(training,start_p=1,start_q=1,max_p=3,max_q=3,m=12,
                  start_P=0,seasonal=True,d=1,D=1,trace=True, error_action= 'ignore',
                  suppress_warnings=True)
model.fit(training)
forecast = model.predict(n_periods=217)


# In[15]:


forecast = pd.DataFrame(forecast,index=test.index,columns=['Predictions'])
#root mean square
rms = np.sqrt(np.mean(np.power((np.array(test['Close'])-np.array(forecast['Predictions'])),2)))
rms


# In[16]:


#plot
plt.figure(figsize=(16,8))
plt.plot(train['Close'])
plt.plot(test['Close'])
plt.plot(forecast['Predictions'])
plt.show()


# In[ ]:


# ARIMA seems to be a really good fit. 


# In[27]:


#LSTM is mainly for: Sequence Prediction
#let's go for LSTM
new_data = df
new_data.index = df.Date
new_data=new_data.drop(columns = ['Date'])


# In[32]:


dataset = new_data.values
dataset = dataset.reshape((len(dataset),1))
#print(dataset)
train = dataset[0:2500,:]
test = dataset[2500:,:]
#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
#print(scaled_data)


# In[35]:


x_train, y_train = [],[]
for i in range(90,len(train)):
    x_train.append(scaled_data[i-90:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


# In[38]:


#create and fit LSTM model
from keras.models import Sequential
model = Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))


model.compile(loss='mean_squared_error', optimizer = 'adam')
model.fit(x_train,y_train,epochs=1,batch_size=1,verbose=2)

#predicting 217 values, using past 90 from the train data
inputs = new_data[len(new_data) - len(test) - 90:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(90,inputs.shape[0]):
    X_test.append(inputs[i-90:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)


# In[39]:


rms = np.sqrt(np.mean(np.power((test-closing_price),2)))
rms


# In[43]:


#plot
train = new_data[:2500]
test = new_data[2500:]
test['Predictions'] = closing_price
plt.figure(figsize=(16,8))
plt.plot(train['Close'])
plt.plot(test[['Close','Predictions']])
plt.show()


# In[ ]:


#inference
#So we can see that LSTM fits best
#The overall conclusion is that these models do give us an estimate of the stock prices but we cannot completely rely on it.

