#!/usr/bin/python3

import time
import os
import RPi.GPIO as GPIO

redpin = 22
greenpin = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(greenpin,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
print("green: ",GPIO.input(greenpin))
GPIO.setup(redpin,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
print("red: ",GPIO.input(redpin))
