# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 02:22:32 2021

@author: Emre Dogan
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 19:31:02 2021

@author: Emre Dogan
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1*3/5)         
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]
 
def average_slope_intercept(image, lines):
    left_fit    = []
    right_fit   = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0: # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # add more weight to longer lines
    if len(left_fit) and len(right_fit):
    ##over-simplified if statement (should give you an idea of why the error occurs)
        left_fit_average  = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line  = make_points(image, left_fit_average)
        right_line = make_points(image, right_fit_average)
        averaged_lines = [left_line, right_line]
        return averaged_lines
def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_image,(x1,y1),(x2,y2),(255,255,0),10)
    return line_image 
    
def canny(image):
    gray=cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    blur=cv.GaussianBlur(gray,(3,3),0)
    canny=cv.Canny(blur,125,175)
    canny2=cv.dilate(canny,(7,7),iterations=7)
    return canny2

def region_of_interest(image):
    height=image.shape[0]
   
    pollygons_2= np.array([[(200,height),(700,height),(500,280)]])
    mask_2=np.zeros_like(image)
    cv.fillPoly(mask_2,pollygons_2,255)
    masked_2=cv.bitwise_and(mask_2,image)
    
    pollygons= np.array([[(0,height),(940,height),(500,280)]])
    mask=np.zeros_like(image)
    cv.fillPoly(mask,pollygons,255)
    masked=cv.bitwise_and(image,mask)
    
    masked_3=masked-masked_2
    return masked_3

src=np.float32([(550,460),(150,720),(1200,720),(770,460)])
dst=np.float32([(100,0),(100,720),(1100,720),(1100,0)])
M=cv.getPerspectiveTransform(src,dst)
M_inv=cv.getPerspectiveTransform(dst,src)
def front_to_top(image):
    size=(1280,720)
    return cv.warpPerspective(image,M,size,flags=cv.INTER_LINEAR)
def top_to_front(image):
    size=(1280,720)
    return cv.warpPerspective(image,M_inv,size,flags=cv.INTER_LINEAR)

image=cv.imread("straight_lines.jpg")
output=top_to_front(image)
display=np.hstack((image,output))
cv.imshow("result",display)
cv.waitKey(0)

"""canny_image=canny(image)
masked=region_of_interest(canny_image)
lines=cv.HoughLinesP(masked,2,np.pi/180,100,np.array([]),minLineLength=1,maxLineGap=5)
averaged_lines=average_slope_intercept(image,lines)
line_image=display_lines(image,averaged_lines)
combo_image=cv.addWeighted(image,0.8,line_image,1,1)
cv.imshow("result",combo_image)
cv.waitKey(0)
"""""""

cap = cv.VideoCapture("test4.mp4")
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        canny_image=canny(frame)
        masked=region_of_interest(canny_image)
        lines=cv.HoughLinesP(masked,2,np.pi/180,100,np.array([]),minLineLength=1,maxLineGap=5)
        averaged_lines=average_slope_intercept(frame,lines)
        line_image=display_lines(frame,averaged_lines)
        combo_image=cv.addWeighted(frame,0.8,line_image,1,1)
        cv.imshow("result", combo_image)
        if cv.waitKey(40) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv.destroyAllWindows()            

        """
   