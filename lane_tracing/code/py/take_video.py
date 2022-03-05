# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 01:53:52 2021

@author: Emre Dogan
"""

import cv2

url="https://10.153.112.29:8080/video"
cam=cv2.VideoCapture(url)
while cam.isOpened():
    ret,frame = cam.read()
    
    if not ret:
        print("nothing shows")
      
   
    frame=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    frame=cv2.Canny(frame, 40, 80)
   
    cv2.imshow("image",frame)
    
    if cv2.waitKey(1)==ord("q"):
        break
cv2.destroyAllWindows()