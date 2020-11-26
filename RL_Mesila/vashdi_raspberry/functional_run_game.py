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



def takePicture():
    _,myImage = cap.read()
    im000 = cv2.medianBlur(myImage,5)
    im000 = im000[150:270,:]
    im000gray  = cv2.cvtColor(im000,cv2.COLOR_BGR2GRAY)
    (thresh,myImageBW) = cv2.threshold(im000gray,180,255,cv2.THRESH_BINARY) 
    im000bw = myImageBW
    #im000rgb = cv2.cvtColor(im000,cv2.COLOR_BGR2RGB)  
    return im000bw,im000

def findCirclesInImg(img000bw,im000color):
    my_dp=3.95
    my_minDist=25
    my_minRadius=25
    my_maxRadius=70
    my_y_upperlimit=75
    my_y_lowerlimit=45
    my_r_upperlimit=60
    my_x=0
    my_y=0
    my_r=0
    my_circles = cv2.HoughCircles(img000bw,cv2.HOUGH_GRADIENT, dp=my_dp,minDist=my_minDist,minRadius=my_minRadius,maxRadius=my_maxRadius)
    if my_circles is not None:
        circles000 = np.round(my_circles[0,:]).astype("int")
            for (x,y,r) in circles000:
                cv2.circle(im000color,(x,y),r,(0,255,0),4)
                cv2.rectangle(im000color,(x-5,y-5),(x+5,y+5),(0,128,255),-1)
                cv2.imshow("im000color",im000color)
                cv2.waitKey(1)
                #print("debug: x,y,r - "+str(x)+","+str(y)+","+str(r))
                if(int(y) > my_y_lowerlimit and int(y) < my_y_upperlimit):
                    if(int(r) < my_r_upperlimit):
                        sub_matrix = im000bw[y-15:y+15,x-15:x+15]
                        #print("debug: avg of sub matrix is: ",sub_matrix.mean())
                        if(sub_matrix.mean() > 200):
                            my_x,my_y,my_r = x,y,r
                            circle2_set = True
                            #print("debug: average is above 200, x,y,z:",str(x),str(y),str(r))
                            cv2.circle(im000color,(x,y),r,(0,255,0),4)
                            cv2.rectangle(im000color,(x-5,y-5),(x+5,y+5),(0,128,255),-1)
                            cv2.imshow("im000color",im000color)
                            cv2.waitKey(1)
    else:
        #print("debug: no circle found set x,y,r = 400,50,50")
        my_x,my_y,my_r = 0,0,0
    return my_X,my_y,my_r

def servoToDegree(servo_num):
    predict_value_to_servo/7.5+0.5
    return 


def degreeToServo(servo_num):
    predict_value_to_servo/7.5+0.5
    return

def numToServo(normal_num):
    # this function get num (Between 0 to 1) and return the value fit for servo and degree
    # i didnt check the angle of slop but assume that it changes between +-30 degrees
    num_to_servo = ((normal_num - 0.5)*2)*3.7 
    predict_degree = (normal_num - 0.5)*60
    return num_to_servo,predict_degree

def servoNumToNormalValue(servo_num):
    # this function get the num we send to the servo motor and return the value between 0 to 1 
    num_to_servo = ((normal_num - 0.5)*2)*3.7
    normal_num = servo_num/7.5 + 0.5
    return normal_num 


############
### INIT ###
############


### INIT SERVO ###
#servoPIN = 17 #empty one for test
gatePIN = 13 

servoPIN = 26 #game servo
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)
p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz
p.start(7.5) # Initialization
print("setting servo to 7.5 [90deg]")
GPIO.setup(gatePIN, GPIO.OUT)
gateservo = GPIO.PWM(gatePIN, 50) # GPIO 17 for PWM with 50Hz
gateservo.start(7)

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
middlepin = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(greenpin,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
#print("green: ",GPIO.input(greenpin))
GPIO.setup(redpin,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
#print("red: ",GPIO.input(redpin))
GPIO.setup(middlepin,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)


### GAME VARS ###
sucess_time = 10
gamma = 0.9
wins = 0
losses = 0 
x_train,y_train = [],[]
predict_value_to_servo,predict_degree = numToServo(0.5) #((0 - 0.5)*2)*3.7
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
            im001bw,im001bgr = takePicture()
            #im001rgb = cv2.cvtColor(im001bgr,cv2.COLOR_BGR2RGB)
            x1,y1,r1 = findCirclesInImg(img001bw,im001bgr)
            if(x1==0 and y1==0 and r1==0):
                x1,y1,r1 = 400,50,50
            print("using x1,y1,r1: ",x1,y1,r1)

        i+=1 
        cRoundCounter += 1
        im002bw,im002bgr = takePicture()
        #im002rgb = cv2.cvtColor(im002bgr,cv2.COLOR_BGR2RGB)
        x2,y2,r2 = findCirclesInImg(img001bw,im002bgr)
        if(x1==0 and y1==0 and r1==0):
            x1,y1,r1 = 400,50,50
        print("using x2,y2,r2: ",x2,y2,r2)
        print("##############################################")


        circleDelta = np.zeros(im001bw.shape)
        deltaX = int(x2-x1)
        vector = np.array((deltaX/640,x2/640,predict_value_to_servo))
        predict_next_value_to_servo_normal = model2.predict(vector.reshape(1,3))
        predict_value_to_servo,predict_degree = numToServo((predict[0][0])
            
        x_train.append(vector)
        y_train.append(predict_next_value_to_servo_normal)

        x1,y1,r1 = x2,y2,r2



        print("debug: predict deltabw: ",predict)
#        if(random() < math.exp(-i/200)):
#            randomnumber = randint(1,20)
#            predict_value_to_servo = (randomnumber-10)/2.7
#            print("debug: choosing random number:",predict_value_to_servo/7.5+0.5)
        if(i%10 == 0):
            print("predict_value_to_servo: ",predict_value_to_servo)
        p.ChangeDutyCycle(7.5 + float(predict_value_to_servo))
        currentImageBW = previousImageBW
        if((GPIO.input(redpin) == 0) or (GPIO.input(greenpin) == 0)):
            cv2.imshow("im002rgb",im002rgb)
            cv2.waitKey(1)
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
            middleCounter=0
            start_time = time.time()
            if((GPIO.input(redpin) == 0)):
                gateservo.ChangeDutyCycle(2)
                print("ball at red gate")
                while((GPIO.input(redpin) == 0)):
                    time.sleep(0.35)
                    dc = dc + 0.25
                    p.ChangeDutyCycle(dc)
                p.ChangeDutyCycle(dc+1)
                time.sleep(1)
                p.ChangeDutyCycle(7.5)
                time.sleep(0.2)
                gateservo.ChangeDutyCycle(7)
            else:
                print("ball at green gate")
                gateservo.ChangeDutyCycle(2)
                while((GPIO.input(greenpin) == 0)):
                    time.sleep(0.35)
                    dc = dc - 0.25
                    p.ChangeDutyCycle(dc)
                p.ChangeDutyCycle(dc-1)
                time.sleep(1)
                p.ChangeDutyCycle(7.5)
                time.sleep(0.2)
                gateservo.ChangeDutyCycle(7)
            time.sleep(0.25)
            cRoundCounter = 0
            gamenumber = gamenumber + 1
            print("debug: starting to play again, game number:",gamenumber)            
            #PSILA
        ### to add check if one of the light switches has been pressed 





except KeyboardInterrupt:
    print("interupted, stoping servo")
    p.stop()
    gateservo.stop()
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

