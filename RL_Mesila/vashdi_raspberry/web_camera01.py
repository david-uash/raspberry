#!/usr/bin/python3.7

from time import sleep
import time
import subprocess 
im001name = "web_im_001.jpg"
im002name = "web_im_002.jpg"
pic_delay = 2
for i in range(pic_delay,0,-1):
  print("take picture in: :"+str(i))
  sleep(1)
start_time = time.time()
subprocess.run(["fswebcam","-r","1280x720","--no-banner","/home/pi/Desktop/im_web_001.jpg"])
print("my program took ",time.time() - start_time," to run") 
for i in range(pic_delay,0,-1):
  print("take picture in: :"+str(i))
  sleep(0.05)

start_time = time.time()
subprocess.run(["fswebcam","-r","1280x720","--no-banner","/home/pi/Desktop/im_web_002.jpg"])
print("my program took ",time.time() - start_time," to run") 
   
