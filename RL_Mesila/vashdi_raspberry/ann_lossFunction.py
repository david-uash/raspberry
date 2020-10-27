
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from keras.layers import Dense
from keras.models import Sequential


model2 = Sequential()
model2.add(Dense(12,input_dim=3,activation='relu'))
model2.add(Dense(8,activation='relu'))
model2.add(Dense(6,activation='relu'))
model2.add(Dense(units=1,activation='sigmoid'))
#model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model2.compile(loss='CategoricalCrossentropy',optimizer='adam',metrics=['accuracy'])
print("the prediction for vector is: ",model2.predict(np.zeros(3).reshape(1,3)))


vector = np.zeros(3).reshape(1,3)
predict = model2.predict(vector)
print("vector",vector)
print("predict",predict)
model2.fit(x=vector,y=predict)

type(vector)
vector2 = np.array((1,1,2)).reshape(1,3)
predict2 = model2.predict(vector2)
print("vector2",vector2)
print("predict2",predict2)
model2.fit(x=vector2,y=predict2)


x_train=[]
y_train=[]
for i in range(1,1000):
  vector2 = np.array((random.randint(1,100),random.randint(1,100),random.randint(1,100))).reshape(1,3)
  predict2 = model2.predict(vector2)
#  print("vector2",vector2)
#  print("predict2",predict2)
  x_train.append(vector2)
  y_train.append(predict2)
  
  
x_train=[]
y_train=[]
for i in range(1,1000):
  vector2 = np.array((random.randint(1,100),random.randint(1,100),random.randint(1,100))).reshape(1,3)
  predict2 = model2.predict(vector2)
#  print("vector2",vector2)
#  print("predict2",predict2)
  x_train.append(vector2)
  y_train.append(predict2)
