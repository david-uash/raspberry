#!/usr/bin/python3

#3.5 - 0 
#7.5 - 90
#12.5 - 180


import RPi.GPIO as GPIO
import time
import sys


#servoPIN = 17 #empty one for test
servoPIN = 13 #game servo 
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz

print("setting servo to 7.5 [90deg]")
p.start(7.5) # Initialization

myinput = input("enter num ([3.5=0deg],[7.5=90deg],[12.5=180deg] - 999 to exit: ")
while(str(myinput) != "999"):
  p.ChangeDutyCycle(float(myinput))
  myinput = input("enter num ([3.5=0deg],[7.5=90deg],[12.5=180deg]) - 999 to exit: ")

p.stop()
GPIO.cleanup()


#########
### limit for game 5.5 - 9.5
#########
