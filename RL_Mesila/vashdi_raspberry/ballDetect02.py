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

camera = PiCamera()
stream = io.BytesIO()
#camera.start_preview()
#sleep(5)
#camera.capture('/home/pi/Desktop/001.jpg')
#camera.stop_preview()

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (36, 0, 0)
greenUpper = (86, 255, 255)
#pts = deque(maxlen=args["buffer"])
pts = deque(maxlen=64)


## if a video path was not supplied, grab the reference
## to the webcam
#if not args.get("video", False):
#	vs = VideoStream(src=0).start()
## otherwise, grab a reference to the video file
#else:
#	vs = cv2.VideoCapture(args["video"])
## allow the camera or video file to warm up
#time.sleep(2.0)

#while(True):
# keep looping
# grab the current frame
#frame = vs.read()

camera.capture(stream,format ='jpeg')

data = np.fromstring(stream.getvalue(),dtype=np.int8)
frame = cv2.imdecode(data,1)

frame = cv2.imread("/home/pi/Desktop/001.jpg")

#plt.imshow(frame)
#plt.show()
# handle the frame from VideoCapture or VideoStream
#frame = frame[1] if args.get("video", False) else frame
# if we are viewing a video and we did not grab a frame,
# then we have reached the end of the video
#if frame is None:
#	break
# resize the frame, blur it, and convert it to the HSV
# color space

frame = imutils.resize(frame, width=600)
blurred = cv2.GaussianBlur(frame, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
# construct a mask for the color "green", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
mask = cv2.inRange(hsv, greenLower, greenUpper)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)

#plt.imshow(mask)
#plt.show()

f,axarr = plt.subplots(2,1)
axarr[0].imshow(frame)
axarr[1].imshow(mask)
plt.show()
