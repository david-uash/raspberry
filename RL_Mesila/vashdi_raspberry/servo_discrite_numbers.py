#!/usr/bin/python3

#3.5 - 0 
#7.5 - 90
#12.5 - 180


import RPi.GPIO as GPIO
import time
import sys


#servoPIN = 17 #empty one for test
servoPIN = 26 #game servo 
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)

p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz

print("setting servo to 7.5 [90deg]")
p.start(7.5) # Initialization

myinput = input("enter num between 0 to 100 (999 to exit) : ")
while(str(myinput) != "999"):
  #p.ChangeDutyCycle(7.5+((float(myinput)-50)/14)) #100 levels
  p.ChangeDutyCycle(7.5+((float(myinput)-10)/2.7))  #20 levels
  myinput = input("enter num between 0 to 100 (999 to exit) : ")

p.stop()
GPIO.cleanup()


#########
### limit for game 5.5 - 9.5
#########
#print(random.randint(1,20))
