#!/usr/bin/python3


import RPi.GPIO as GPIO
import time 

GPIO.setmode(GPIO.BOARD)
GPIO.setup(33,GPIO.OUT)
p = GPIO.PWM(33,50)
print("start servo")
p.start(7.5)

print("change dutycycle to 30")
p.ChangeDutyCycle(4)
print("sleep 3 sec")
time.sleep(3)
print("change dutycycle to 60")
p.ChangeDutyCycle(10)
print("sleep 3 sec")
time.sleep(3)
print("exit")
p.stop()
GPIO.cleanup()

