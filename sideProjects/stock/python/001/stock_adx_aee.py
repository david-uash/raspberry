#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
from sklearn.metrics import confusion_matrix


# In[2]:


aee_csv = pd.read_csv("/home/vashdi/stock/python/001/aee.txt")
aee_csv


# In[3]:


aee_csv_data = aee_csv[["Open","High","Low","Close","Volume"]]
aee_csv_data


# In[4]:


aee_csv_data_diff = aee_csv_data.diff()
aee_csv_data_diff


# In[5]:


VolumMean=aee_csv_data["Volume"].mean()


# In[6]:


aee_csv_data_vol_normvol = aee_csv_data_diff[["Volume"]].div(VolumMean)
aee_csv_data_diff_stack = aee_csv_data_diff[["Open","High","Low","Close"]]
aee_csv_data_diff_stack["NormVol"]  = aee_csv_data_vol_normvol
aee_csv_data_diff_stack


# In[7]:


x_arr = aee_csv_data_diff_stack


# In[8]:


df_open_close = aee_csv[["Open","Close"]]
df_open_close


# In[9]:


t = aee_csv_data_diff_stack['Close'] > 0
aee_csv_data_diff_stack['CtgtCn'] = t 
aee_csv_data_diff_stack


# In[10]:


y_arr = t


# In[11]:


import keras
from keras.models import Sequential
from keras.layers import Convolution2D 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[12]:


onerow = np.array(aee_csv_data_diff_stack.head(2).tail(1))
print("onerow:",onerow)
print("onerow.shape:",onerow.shape)


# In[13]:


test = np.zeros((1,6,1))
test


# In[14]:


print("onerow.shape[0]",onerow.shape[0])
print("onerow.shape[1]",onerow.shape[1])
inputshape = np.zeros([onerow.shape[0],onerow.shape[1],1])
print("inputshape.shape",inputshape.shape)
                      


# In[15]:


###########
### CNN ###
###########
inputshape = np.zeros([onerow.shape[0],onerow.shape[1],1]) #(1,6,1) 
model2 = Sequential()
model2.add(Dense(12,input_dim=5,activation='relu'))
model2.add(Dense(8,activation='relu'))
model2.add(Dense(6,activation='relu'))
model2.add(Dense(units=1,activation='sigmoid'))
model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model2.predict(np.zeros([onerow.shape[0],onerow.shape[1]-1])))
#model2.predict(a) a.shape = (1, 120, 640, 1)


# In[16]:


x_arr


# In[17]:


bad_x = x_arr.index.isin([0])
x_arr[~bad_x]
X_Arr = x_arr[~bad_x][["Open","High","Low","Close","NormVol"]]
Y_Arr = x_arr[~bad_x][["CtgtCn"]]


# In[18]:


X_Arr


# In[19]:


Y_Arr


# In[20]:


a = np.array(X_Arr.head(1))
a.reshape(1,5)


# In[21]:


model2.predict(a.reshape(1,5))


# In[22]:


#model2.fit(x=np.vstack(X_Arr.to_numpy()),y=np.vstack(Y_Arr.to_numpy()))


# In[23]:


#model2.fit(x=np.vstack(X_Arr.to_numpy()),y=np.vstack(Y_Arr.to_numpy()),batch_size=10,epochs=100)


# In[24]:


bad_x = x_arr.index.isin([0])
bigArr= x_arr[~bad_x]
msk = np.random.rand(len(bigArr)) < 0.8
Arr_train=bigArr[msk]
Arr_test=bigArr[~msk]
X_Arr_train = Arr_train[["Open","High","Low","Close","NormVol"]]
Y_Arr_train = Arr_train[["CtgtCn"]]
X_Arr_test = Arr_test[["Open","High","Low","Close","NormVol"]]
Y_Arr_test = Arr_test[["CtgtCn"]]
print(len(Y_Arr_test))
print(len(Y_Arr_train))

#model2.fit(x=np.vstack(X_Arr.to_numpy()),y=np.vstack(Y_Arr.to_numpy()))


# In[25]:


model2.fit(x=np.vstack(X_Arr_train.to_numpy()),y=np.vstack(Y_Arr_train.to_numpy()),batch_size=10,epochs=100)


# In[26]:


Y_Arr_predict = model2.predict(X_Arr_test)
Y_Arr_predict = Y_Arr_predict > 0.5


# In[27]:


print(type(Y_Arr_test.to_numpy()))
print(type(Y_Arr_predict))
y_test=Y_Arr_test.to_numpy()
cm = confusion_matrix(y_test,Y_Arr_predict)
print(cm)


# In[33]:


adx_csv = pd.read_csv("/home/vashdi/stock/python/001/adx.txt")
adx_csv
adx_csv_data = adx_csv[["Open","High","Low","Close","Volume"]]
adx_csv_data
adx_csv_data_diff = adx_csv_data.diff()
adx_csv_data_diff
VolumAdxMean=adx_csv_data["Volume"].mean()
adx_csv_data_vol_normvol = adx_csv_data_diff[["Volume"]].div(VolumMean)
adx_csv_data_diff_stack = adx_csv_data_diff[["Open","High","Low","Close"]]
adx_csv_data_diff_stack["NormVol"]  = adx_csv_data_vol_normvol
adx_csv_data_diff_stack
#######################
x_arr_adx = adx_csv_data_diff_stack

adx_df_open_close = adx_csv[["Open","Close"]]
t_adx = adx_csv_data_diff_stack['Close'] > 0
adx_csv_data_diff_stack['CtgtCn'] = t_adx
adx_csv_data_diff_stack
y_arr_adx = t_adx

#adx_csv_data_diff_stack

bad_x = x_arr_adx.index.isin([0])

X_Arr_adx = x_arr_adx[["Open","High","Low","Close","NormVol"]]
Y_Arr_adx = x_arr_adx[["CtgtCn"]]


Y_Arr_predict_adx = model2.predict(X_Arr_adx)
Y_Arr_predict_adx = Y_Arr_predict_adx > 0.5



# In[34]:


#y_test=Y_Arr_test.to_numpy()
cm = confusion_matrix(Y_Arr_adx.to_numpy(),Y_Arr_predict_adx)
print(cm)

