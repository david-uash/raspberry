#!/usr/bin/python3.7

from picamera import PiCamera
from time import sleep

from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

camera = PiCamera()

camera.start_preview()
sleep(5)
#camera.capture('/home/pi/Desktop/001.jpg')



camera.stop_preview()
