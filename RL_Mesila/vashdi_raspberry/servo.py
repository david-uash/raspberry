#!/usr/bin/python3

#3.5 - 0 
#7.5 - 90
#12.5 - 180


import RPi.GPIO as GPIO
import time
import sys

servoPIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz

p.start(2.5) # Initialization

myinput = input("enter num ([3.5=0deg],[7.5=90deg],[12.5=180deg] : ")
while(str(myinput) != "999"):
  p.ChangeDutyCycle(float(myinput))
  myinput = input("enter num: ")

p.stop()
GPIO.cleanup()



