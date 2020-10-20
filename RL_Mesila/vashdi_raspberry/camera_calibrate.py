#!/usr/bin/python3.7

from time import sleep
import time
import subprocess 
import cv2
import matplotlib.pyplot as plt


im001name = "web_im_001.jpg"
im002name = "web_im_002.jpg"
pic_delay = 2
for i in range(pic_delay,0,-1):
  print("take picture in: :"+str(i))
  sleep(1)
start_time = time.time()
subprocess.run(["fswebcam","-r","1280x720","--no-banner","/home/pi/Desktop/im_web_001.jpg"])
sleep(0.5)
subprocess.run(["fswebcam","-r","1280x720","--no-banner","/home/pi/Desktop/im_web_001.jpg"])
print("my program took ",time.time() - start_time," to run") 

### add outer line ###
start_time = time.time()
im001 = cv2.imread("/home/pi/Desktop/im_web_001.jpg")
im001 = cv2.medianBlur(im001,5)
x1,y1 = 0,150
x2,y2 = 640,150
x3,y3 = 0,270
x4,y4 = 640,270
width,height =640,480 
line_thickness = 2
cv2.line(im001,(x1,y1),(x2,y2),(0,255,0),thickness=line_thickness)
cv2.line(im001,(x3,y3),(x4,y4),(0,255,0),thickness=line_thickness)
### add inner line ###
x1,y1 = 0,195
x2,y2 = 640,195
x3,y3 = 0,225
x4,y4 = 640,225
width,height =640,480 
line_thickness = 2
cv2.line(im001,(x1,y1),(x2,y2),(0,0,255),thickness=line_thickness)
cv2.line(im001,(x3,y3),(x4,y4),(0,0,255),thickness=line_thickness)

#im001 = im001[150:270,:]
#im002 = im002[150:270,:]
im001gray  = cv2.cvtColor(im001,cv2.COLOR_BGR2GRAY)
(thresh,im001bw) = cv2.threshold(im001gray,180,255,cv2.THRESH_BINARY) #original threshold = 127 (now 180)
im001rgb = cv2.cvtColor(im001,cv2.COLOR_BGR2RGB)



fig = plt.figure()#figsize=(30,30))
plt1 = fig.add_subplot(3,1,1)
plt.imshow(im001rgb)
plt3 = fig.add_subplot(3,1,2)
plt.imshow(im001gray,cmap='Greys')
plt5 = fig.add_subplot(3,1,3)
plt.imshow(im001bw,cmap='Greys_r')
#plt9 = fig.add_subplot(5,2,9)
#plt.imshow(circleDelta_reshape,cmap='Greys_r')


plt1.title.set_text("im001rgb")
plt3.title.set_text("im001gray")
plt5.title.set_text("im001bw")
plt.show()
#

