#!/usr/bin/python3

import time
import os
import RPi.GPIO as GPIO
import Jetson.GPIO as readgpio
#redpin = 22
greenpin = 7
#middlepin = 4
readgpio.setmode(GPIO.BOARD)
readgpio.setup(greenpin,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
print("green: ",GPIO.input(greenpin))
#GPIO.setup(redpin,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
#print("red: ",GPIO.input(redpin))
#GPIO.setup(middlepin,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
#print("middle: ",GPIO.input(middlepin))

try:
    while True:
        os.system('clear')
        print("green: ",GPIO.input(greenpin))
#        print("red: ",GPIO.input(redpin))
#        print("middle: ",GPIO.input(middlepin))
        time.sleep(0.2)
except KeyboardInterrupt:
    print("end")

