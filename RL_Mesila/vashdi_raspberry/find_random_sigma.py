#!/usr/bin/python3.7
import numpy as np

mu = 0
sigma = 0.2


sigma = input("enter sigma (type 999 to exit) : ")
while(str(sigma) != "999"):
  a = np.random.normal(mu,float(sigma),1000000)
  print("max num is: ",a.max()," min num is: ",a.min())
  sigma = input("enter sigma (type 999 to exit) : ")





