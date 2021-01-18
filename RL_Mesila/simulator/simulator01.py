##!/jupyter

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
##################################################
pole_length = 0.3 #[m]
angle=0
time_interval = 0.2 #[sec]
g=9.8 #[m/s^2]
ball_mass = 0.02 #[kg]
ball_radius = 0.025 #[m]
friction_coefficient = 0.2
v0 = 0
w0 = 0
x0 = 0 #[m]
I_ball  = (2/3)*ball_mass*(ball_radius**2)
###################################################
degree=10
print("current degree:",degree)
angle=math.pi*(degree/180)
angular_acceleration = (3/2)*(g/ball_radius)*friction_coefficient*math.cos(angle)
a=(ball_radius*friction_coefficient*ball_mass*g*math.cos(angle))/((2/3)*ball_mass*ball_radius*ball_radius)
#after [time_interval]
if(angle>=0):
  angle_sign = 1
else:
  angle_sign = -1
current_w = w0 + angle_sign*angular_acceleration*time_interval 
current_theta = w0*time_interval + (1/2)*angle_sign*angular_acceleration*(time_interval**2)
current_x = x0 + ball_radius*current_theta

w0 = current_w
#print("after",time_interval,"[sec], at angle=",angle*180/math.pi,"[deg], the ball moved to ",current_x)

normalize_x = 300-current_x*300/1.5
print("current_x",current_x)
print("normalize_x",normalize_x)
height,width=240,640
length = 610
image = np.ones((height, width)) * 255
#draw line
#x1, y1 = 10, int(height/2)
x1, y1 = int(width/2), int(height/2)
x2 = x1+int(length/2*math.cos(angle))
y2 = y1+int(length/2*math.sin(angle))
x1,y1 = (x1-int(length/2*math.cos(angle))),(y1-int(length/2*math.sin(angle)))
line_thickness = 2
cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)

M = (y2-y1)/(x2-x1)
y_center_of_circle = M*normalize_x + y1 -M*x1

#draw circle
radius = 10
center_coordinates = (int(normalize_x),int(y_center_of_circle+radius))
color = (0, 255, 0)
thickness = 2
image = cv2.circle(image, center_coordinates, radius, color, thickness)


fig = plt.figure()
plt1 = fig.add_subplot(1,1,1)
plt.imshow(image,cmap='Greys_r',origin='lower')

