#!/usr/bin/python3.7

from picamera import PiCamera
from time import sleep

import matplotlib.pyplot as plt
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import io

#camera = PiCamera()
#stream = io.BytesIO()
#camera.start_preview()
#sleep(5)
#camera.capture('/home/pi/Desktop/001.jpg')
#camera.stop_preview()
frame = cv2.imread("/home/pi/Desktop/001.jpg")
output = frame.copy()
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT, 1.2,100)
if circles is not None:
    circles = np.round(circles[0,:]).astype("int")
    for (x,y,r) in circles:
        cv2.circle(output,(x,y),r,(0,255,0),4)
        cv2.rectangle(output,(x-5,y-5),(x+5,y+5),(0,128,255),-1)



#plt.imshow(frame)
#plt.show()


f,axarr = plt.subplots(2,1)
axarr[0].imshow(frame)
axarr[1].imshow(output)
plt.show()
