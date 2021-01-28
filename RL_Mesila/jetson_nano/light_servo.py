#!/usr/bin/python3

import time
import os
import RPi.GPIO as GPIO
import Jetson.GPIO as readgpio

GPIO.setmode(GPIO.BOARD)
GPIO.setup(33,GPIO.OUT)
p = GPIO.PWM(33,50)
print("start servo")
p.start(7.5)

greenpin = 7
readgpio.setmode(GPIO.BOARD)
readgpio.setup(greenpin,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
print("green: ",GPIO.input(greenpin))

try:
    while True:
        os.system('clear')
        lights = GPIO.input(greenpin)
        print("green: ",lights)
        if(lights==0):
            print("change dutycycle to 30")
            p.ChangeDutyCycle(4)
            print("sleep 2 sec")
            time.sleep(2)
        else:
            print("change dutycycle to 60")
            p.ChangeDutyCycle(10)
            print("sleep 2 sec")
            time.sleep(2)
        time.sleep(0.2)
except KeyboardInterrupt:
    print("end")
    p.stop()
    GPIO.cleanup()

