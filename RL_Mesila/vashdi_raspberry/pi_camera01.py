#!/usr/bin/python3.7

from picamera import PiCamera
from time import sleep

camera = PiCamera()

camera.start_preview()
sleep(10)
camera.capture('/home/pi/Desktop/001.jpg')
camera.stop_preview()
