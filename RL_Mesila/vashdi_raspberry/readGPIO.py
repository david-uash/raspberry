#!/usr/bin/python3

import time
import os
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setup(27,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
print(GPIO.input(27))
GPIO.setup(22,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
print(GPIO.input(22))
