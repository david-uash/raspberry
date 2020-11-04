#!/usr/bin/python3.7

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from random import random
from random import randint
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
vector = np.array((0,0,0))
###########
### CNN ###
###########
inputshape = vector.shape
model2 = Sequential()
model2.add(Dense(12,input_dim=3,activation='relu'))
model2.add(Dense(8,activation='relu'))
model2.add(Dense(6,activation='relu'))
model2.add(Dense(units=1,activation='sigmoid'))
#model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#model2.compile(loss='CategoricalCrossentropy',optimizer='adam',metrics=['accuracy'])
model2.compile(loss='mean_squared_error',optimizer='adam',metrics=['mse'])
print("the prediction for vector is: ",model2.predict(vector.reshape(1,3)))
print("the prediction for vector is: ",model2.predict(np.zeros(3).reshape(1,3)))
#model2.predict(np.zeros((1,3)))
############################################




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
predict_value_to_servo = ((0 - 0.5)*2)*3.7
i_counter = 0
cRoundCounter = 0
lossesCounter = 0
winsCounter = 0
gamenumber=0
i=0
circle1_set,circle2_set = False,False
try:
    while True:
        print("i_counter:",i)
        if(cRoundCounter == 0):
            _,previousImage = cap.read()
            im001 = cv2.medianBlur(previousImage,5)
            im001 = im001[150:270,:]
            im001gray  = cv2.cvtColor(im001,cv2.COLOR_BGR2GRAY)
            (thresh,previousImageBW) = cv2.threshold(im001gray,180,255,cv2.THRESH_BINARY) #original threshold = 127 (now 180)
            im001bw = previousImageBW
            #im001rgb = cv2.cvtColor(im001,cv2.COLOR_BGR2RGB)
            circles1 = cv2.HoughCircles(im001bw,cv2.HOUGH_GRADIENT, dp=3.95,minDist=25,minRadius=25,maxRadius=70)
            if circles1 is not None:
                #print("debug: found circle in im001bw")
                circles1 = np.round(circles1[0,:]).astype("int")
                for (x,y,r) in circles1:
                    #print("debug: x,y,r - "+str(x)+","+str(y)+","+str(r))
                    if(int(y) > 45 and int(y) < 75):
                        if(int(r) < 60):
                             sub_matrix = im001bw[y-15:y+15,x-15:x+15]
                             #print("debug: avg of sub matrix for x,y,r",x,y,r," is: ",sub_matrix.mean())
                             if(sub_matrix.mean() > 200):
                                x1,y1,r1 = x,y,r
                                circle1_set = True
                                print("debug: average is above 200, x,y,z:",str(x1),str(y1),str(r1))
                                #cv2.circle(im001rgb,(x,y),r,(0,255,0),4)
                                #cv2.rectangle(im001rgb,(x-5,y-5),(x+5,y+5),(0,128,255),-1)
            else:
                #print("debug: no circle found set x,y,r = 400,50,50")
                x1,y1,r1 = 400,50,50
            if(circle1_set != True):
                x1,y1,r1 = 400,50,50
                #print("debug: no good circle found, using x1,y1,r1: ",x1,y1,r1)
            circle1_set = False
            print("using x1,y1,r1: ",x1,y1,r1)
            print("##############################################")

        i+=1 
        cRoundCounter += 1
        _,currentImage = cap.read() 
        im002 = cv2.medianBlur(currentImage,5)
        im002 = im002[150:270,:]
        im002gray  = cv2.cvtColor(im002,cv2.COLOR_BGR2GRAY)
        (thresh,correntImageBW) = cv2.threshold(im002gray,180,255,cv2.THRESH_BINARY)
        im002bw = correntImageBW 
        circles2 = cv2.HoughCircles(im002bw,cv2.HOUGH_GRADIENT, dp=3.95,minDist=25,minRadius=25,maxRadius=70)
        if circles2 is not None:
            #print("debug: found circle in im002bw")
            circles2 = np.round(circles2[0,:]).astype("int")
            for (x,y,r) in circles2:
                #print("debug: x,y,r - "+str(x)+","+str(y)+","+str(r))
                if(int(y) > 45 and int(y) < 75):
                    if(int(r) < 60):
                        sub_matrix = im002bw[y-15:y+15,x-15:x+15]
                        #print("debug: avg of sub matrix is: ",sub_matrix.mean())
                        if(sub_matrix.mean() > 200):
                            x2,y2,r2 = x,y,r
                            circle2_set = True
                            print("debug: average is above 200, x,y,z:",str(x2),str(y2),str(r2))
                            #cv2.circle(im002rgb,(x,y),r,(0,255,0),4)
                            #cv2.rectangle(im002rgb,(x-5,y-5),(x+5,y+5),(0,128,255),-1)
        else:
            #print("debug: no circle found set x,y,r same as before (x1,y1,r1)")
            x2,y2,r2 = x1,y1,r1
        if(circle2_set != True):
            x2,y2,r2 = x1,y1,r1
            #print("debug: no good circle found, using x1,y1,r1: ",x1,y1,r1)
            circle2_set = False
        print("using x2,y2,r2: ",x2,y2,r2)
        print("##############################################")
        circleDelta = np.zeros(im001bw.shape)
        deltaX = int(x2-x1)
        print("debug: ### delta x2-x1 = ",str(deltaX)," ###")
        #print("debug: normal value of delta x (x/640): ",float(deltaX/640))
        #circleDelta[y2-r2:y2+r2,x2-r2:x2+r2] = im002bw[y2-r2:y2+r2,x2-r2:x2+r2] - im001bw[y2-r2:y2+r2,x2-r2:x2+r2]
        
        vector = np.array((deltaX/640,x2/640,predict_value_to_servo/7.5+0.5))
        predict = model2.predict(vector.reshape(1,3))
        predict_value_to_servo = ((predict[0][0] - 0.5)*2)*3.7
            
        x_train.append(vector)
        y_train.append(predict)

        x1,y1,r1 = x2,y2,r2



        print("debug: predict deltabw: ",predict)
        if(random() < math.exp(-i/200)):
            randomnumber = randint(1,20)
            predict_value_to_servo = (randomnumber-10)/2.7
            print("debug: choosing random number:",predict_value_to_servo/7.5+0.5)
        if(i%10 == 0):
            print("predict_value_to_servo: ",predict_value_to_servo)
        p.ChangeDutyCycle(7.5 + float(predict_value_to_servo))
        currentImageBW = previousImageBW
        if((GPIO.input(redpin) == 0) or (GPIO.input(greenpin) == 0)):
            print("num of steps in this round : ",cRoundCounter)
            lossesCounter += 1 
            #print("bring ball back to middle")
            if(cRoundCounter < 3):
                for j in range(1,len(y_train),1): 
                    y_train[len(y_train)-j] *= (-1)
            elif(cRoundCounter < 10):
                for j in range(1,2,1): #start at 0 until 3 in steps of 1 (0,1,2) 
                    y_train[len(y_train)-j] *= (-1)
            else:
                for j in range(1,4,1):
                    y_train[len(y_train)-j] *= (-1)

            print("adjasting model")
            start_time = time.time()
            model2.fit(x=np.vstack(x_train),y=np.vstack(y_train))
            #print("adjasting model took: ",time.time() - start_time)
            x_train,y_train = [],[]
            time.sleep(2)
            dc = 7.5 + float(predict_value_to_servo)
            if((GPIO.input(redpin) == 0)):
                print("ball at red gate")
                while((GPIO.input(redpin) == 0)):
                    time.sleep(0.35)
                    dc = dc + 0.1
                    p.ChangeDutyCycle(dc)
                time.sleep(0.35)
                p.ChangeDutyCycle(7.7)
                #time.sleep(0.1)
                #p.ChangeDutyCycle(4.5)
                #time.sleep(1.8)
                #p.ChangeDutyCycle(9)
                #time.sleep(0.77)
                #p.ChangeDutyCycle(6.5)
            else:
                print("ball at green gate")
                while((GPIO.input(greenpin) == 0)):
                    time.sleep(0.35)
                    dc = dc - 0.1
                    p.ChangeDutyCycle(dc)
                p.ChangeDutyCycle(7.85)
                #time.sleep(0.1)
                #p.ChangeDutyCycle(9.5)
                #time.sleep(1.8)
                #p.ChangeDutyCycle(4.5)
                #time.sleep(0.67)
                #p.ChangeDutyCycle(8.3)
            time.sleep(0.15)
            cRoundCounter = 0
            gamenumber = gamenumber + 1
            print("debug: starting to play again, game number:",gamenumber)            
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

