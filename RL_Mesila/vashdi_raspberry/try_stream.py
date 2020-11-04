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

#import keras
#from keras.models import Sequential
#from keras.layers import Convolution2D 
#from keras.layers import Conv2D
#from keras.layers import MaxPooling2D
#from keras.layers import Flatten
#from keras.layers import Dense

#import RPi.GPIO as GPIO





############
### INIT ###
############


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

while True:
    _,pic = cap.read()
    cv2.imshow("image",pic)
    time.sleep(1)
    cv2.waitKey(1)

    

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
                                cv2.circle(im001rgb,(x,y),r,(0,255,0),4)
                                cv2.rectangle(im001rgb,(x-5,y-5),(x+5,y+5),(0,128,255),-1)
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
                            cv2.circle(im002rgb,(x,y),r,(0,255,0),4)
                            cv2.rectangle(im002rgb,(x-5,y-5),(x+5,y+5),(0,128,255),-1)
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
        
#        image.show




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

