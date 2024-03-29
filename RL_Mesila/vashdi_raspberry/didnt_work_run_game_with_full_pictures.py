#!/usr/bin/python3.7

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from random import random
import time
import math

import keras
from keras.models import Sequential
from keras.layers import Convolution2D 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

import RPi.GPIO as GPIO





############
### INIT ###
############


### INIT SERVO ###
#servoPIN = 17 #empty one for test
servoPIN = 26 #game servo
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)
p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz
print("setting servo to 7.5 [90deg]")
p.start(7.5) # Initialization






### INIT CAMERA ###
###################################
### Read from Video (Streaming) ###
# cap = cv2.VideoCapture(0)        
# _,frame = cap.read()             
# plt.imshow(frame)
# plt.show()
###################################
cap = cv2.VideoCapture(0)
_,initpic = cap.read()
for i in range(3,0,-1):
  print("take picture in: :"+str(i))
  time.sleep(1)

### INIT CNN ###
###########
### CNN ###
###########

print("init cnn input shape to (120,640)")
circleDeltashape = (120,640)
circleDeltaAsInputShape = np.zeros([circleDeltashape[0],circleDeltashape[1],1])
inputshape = circleDeltaAsInputShape.shape  #(120,640,1) 
model2 = Sequential()
model2.add(Conv2D(128,kernel_size=(2,2),input_shape=inputshape))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Conv2D(32,(3,3),activation='relu'))
model2.add(Flatten())
model2.add(Dense(units=32,activation='relu'))
model2.add(Dense(units=1,activation='sigmoid'))
model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model2.predict(np.zeros([1,circleDeltashape[0],circleDeltashape[1],1])/255))
#model2.predict(a) a.shape = (1, 120, 640, 1)
###########



### INIT GPIO ###
redpin = 22
greenpin = 27
GPIO.setmode(GPIO.BCM)
GPIO.setup(greenpin,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
#print("green: ",GPIO.input(greenpin))
GPIO.setup(redpin,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
#print("red: ",GPIO.input(redpin))



### GAME VARS ###
sucess_time = 10
gamma = 0.9
wins = 0
losses = 0 
x_train,y_train = [],[]
i_counter = 0
cRoundCounter = 0
lossesCounter = 0
winsCounter = 0
try:
    _,previousImage = cap.read() 
    im001 = cv2.medianBlur(previousImage,5)
    im001 = im001[150:270,:]
    im001gray  = cv2.cvtColor(im001,cv2.COLOR_BGR2GRAY)
    (thresh,previousImageBW) = cv2.threshold(im001gray,180,255,cv2.THRESH_BINARY) #original threshold = 127 (now 180)
    #im001rgb = cv2.cvtColor(im001,cv2.COLOR_BGR2RGB)
    while True:
        i+=1 
        cRoundCounter += 1 
        _,currentImage = cap.read() 
        im002 = cv2.medianBlur(currentImage,5)
        im002 = im002[150:270,:]
        im002gray  = cv2.cvtColor(im002,cv2.COLOR_BGR2GRAY)
        (thresh,correntImageBW) = cv2.threshold(im002gray,180,255,cv2.THRESH_BINARY)
        #im002rgb = cv2.cvtColor(im002,cv2.COLOR_BGR2RGB)
        if(previousImageBW is not None): 
          deltabw = (correntImageBW - previousImageBW)/255
        else:
            deltabw = np.zeros(correntImageBW.shpae)/255
        predict = model2.predict(deltabw.reshape([1,circleDeltashape[0],circleDeltashape[1],1])) 
        predict_value_to_servo = ((predict - 0.5)*2)*2
        print("predict deltabw: ",predict)
        if(random() < math.exp(-i/200)):
            print("choosing random number")
            predict_value_to_servo = (random()-0.5)*2*2
        if(i%10 == 0):
            print("predict_value_to_servo: ",predict_value_to_servo)
        p.ChangeDutyCycle(7.5 + float(predict_value_to_servo))
        x_train.append(deltabw.reshape([1,circleDeltashape[0],circleDeltashape[1],1]))
        y_train.append(predict_value_to_servo)
        currentImageBW = previousImageBW
        if((GPIO.input(redpin) == 0) or (GPIO.input(greenpin) == 0)):
            print("num of steps in this round : ",cRoundCounter)
            lossesCounter += 1 
            print("bring ball back to middle")
            if(cRoundCounter < 3):
                for j in range(1,len(y_train),1): 
                    y_train[len(y_train)-j] *= (-1)
            elif(cRoundCounter < 10):
                for j in range(1,3,1): #start at 0 until 3 in steps of 1 (0,1,2) 
                    y_train[len(y_train)-j] *= (-1)
            else:
                for j in range(1,6,1):
                    y_train[len(y_train)-j] *= (-1)

            print("adjasting model")
            start_time = time.time()
            model2.fit(x=np.vstack(x_train),y=np.vstack(y_train))
            print("adjasting model took: ",time.time() - start_time)
            x_train,y_train = [],[]
            if((GPIO.input(redpin) == 0)):
                print("ball at red gate")
                p.ChangeDutyCycle(8.5)
                time.sleep(1.5)
                p.ChangeDutyCycle(7.2)
            else:
                print("ball at green gate")
                p.ChangeDutyCycle(6.5)
                time.sleep(1.5)
                p.ChangeDutyCycle(7.8)

            cRoundCounter = 0
            
            #PSILA
        ### to add check if one of the light switches has been pressed 





except KeyboardInterrupt:
    print("interupted, stoping servo")
    p.stop()
    GPIO.cleanup()


start_time = time.time()
#im001 = cv2.imread("/home/pi/Desktop/im_web_001.jpg")
#im002 = cv2.imread("/home/pi/Desktop/im_web_002.jpg")
_,im001 = cap.read() #Read 001 2 times - dont know why, but the first image is very bad (maybe the camera light is still adjasting)
_,im001 = cap.read()
time.sleep(0.5)
_,im002 = cap.read()
print("take 2 pic + 0.5sec took: ",time.time() - start_time)
im001 = cv2.medianBlur(im001,5)
im002 = cv2.medianBlur(im002,5)
im001 = im001[150:270,:]
im002 = im002[150:270,:]
im001gray  = cv2.cvtColor(im001,cv2.COLOR_BGR2GRAY)
im002gray  = cv2.cvtColor(im002,cv2.COLOR_BGR2GRAY)
(thresh,im001bw) = cv2.threshold(im001gray,180,255,cv2.THRESH_BINARY) #original threshold = 127 (now 180)
(thresh,im002bw) = cv2.threshold(im002gray,180,255,cv2.THRESH_BINARY)
im001rgb = cv2.cvtColor(im001,cv2.COLOR_BGR2RGB)
im002rgb = cv2.cvtColor(im002,cv2.COLOR_BGR2RGB)
deltabw = im002bw - im001bw

circles1 = cv2.HoughCircles(im001bw,cv2.HOUGH_GRADIENT, dp=3.95,minDist=25,minRadius=30,maxRadius=70)
if circles1 is not None:
    print("found circle")
    circles1 = np.round(circles1[0,:]).astype("int")
    for (x,y,r) in circles1:
        print("x,y,r - "+str(x)+","+str(y)+","+str(r))
        cv2.circle(im001rgb,(x,y),r,(0,255,0),4)
        cv2.rectangle(im001rgb,(x-5,y-5),(x+5,y+5),(0,128,255),-1)
else:
    print("no circle found")


circles2 = cv2.HoughCircles(im002bw,cv2.HOUGH_GRADIENT, dp=3.95,minDist=25,minRadius=30,maxRadius=70)
if circles2 is not None:
    print("found circle")
    circles2 = np.round(circles2[0,:]).astype("int")
    for (x,y,r) in circles2:
        print("x,y,r - "+str(x)+","+str(y)+","+str(r))
        cv2.circle(im002rgb,(x,y),r,(0,255,0),4)
        cv2.rectangle(im002rgb,(x-5,y-5),(x+5,y+5),(0,128,255),-1)
else:
    print("no circle found")

circleDelta = np.zeros(im001bw.shape)
if((circles1 is not None) and (circles2 is not None)):
  circleDelta = np.zeros(im001bw.shape)
  for x,y,r in circles2: #np.concatenate((circles1,circles2)):
      print("x,y,r - "+str(x)+","+str(y)+","+str(r))
      circleDelta[y-r:y+r,x-r:x+r] = im002bw[y-r:y+r,x-r:x+r] - im001bw[y-r:y+r,x-r:x+r]

print("my program took ",time.time() - start_time," to run")
############################

###########
### CNN ###
###########
print(circleDelta.shape)
circleDeltashape = circleDelta.shape
circleDeltaAsInputShape = np.zeros([circleDeltashape[0],circleDeltashape[1],1])
inputshape = circleDeltaAsInputShape.shape  #(120,640,1) #need to find how to set the size by the shape of circleDelta - (and add 1 to the shape !!!)
model2 = Sequential()
model2.add(Conv2D(128,kernel_size=(2,2),input_shape=inputshape))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Conv2D(32,(3,3),activation='relu'))
model2.add(Flatten())
model2.add(Dense(units=32,activation='relu'))
model2.add(Dense(units=1,activation='sigmoid'))
model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model2.predict(circleDelta.reshape([1,circleDeltashape[0],circleDeltashape[1],1])))
#model2.predict(a) a.shape = (1, 120, 640, 1)
###########

############################
fig = plt.figure()#figsize=(30,30))
plt1 = fig.add_subplot(5,2,1)
plt.imshow(im001rgb)
plt2 =fig.add_subplot(5,2,2)
plt.imshow(im002rgb)
plt3 = fig.add_subplot(5,2,3)
plt.imshow(im001gray,cmap='Greys')
plt4 = fig.add_subplot(5,2,4)
plt.imshow(im002gray,cmap='Greys')
plt5 = fig.add_subplot(5,2,5)
plt.imshow(im001bw,cmap='Greys_r')
plt6 = fig.add_subplot(5,2,6)
plt.imshow(im002bw,cmap='Greys_r')
plt7 = fig.add_subplot(5,2,7)
plt.imshow(deltabw,cmap='Greys_r')
plt8 = fig.add_subplot(5,2,8)
plt.imshow(circleDelta,cmap='Greys_r')
#plt9 = fig.add_subplot(5,2,9)
#plt.imshow(circleDelta_reshape,cmap='Greys_r')


plt1.title.set_text("im001rgb")
plt2.title.set_text("im002rgb")
plt3.title.set_text("im001gray")
plt4.title.set_text("im002gray")
plt5.title.set_text("im001bw")
plt6.title.set_text("im002bw")
plt7.title.set_text("deltabw")
plt8.title.set_text("circleDelta")
plt.show()
#
#
#

p.stop()
GPIO.cleanup()

