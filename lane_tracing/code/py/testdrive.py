# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 22:28:50 2021

@author: Emre Dogan
"""

import cv2 as cv
import numpy as np
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
def display_lines(img,lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image 
    
def canny(image):
    gray=cv.cvtColor(image,cv.COLOR_RGB2GRAY)
    blur=cv.GaussianBlur(gray,(3,3),0)
    canny=cv.Canny(blur,125,175)
    return canny

def region_of_interest(image):
    height=image.shape[0]
    pollygons= np.array([[(130,height),(890,height),(480,305)]])
    mask=np.zeros_like(image)
    cv.fillPoly(mask,pollygons,255)
    masked=cv.bitwise_and(image,mask)
    return masked


image=cv.imread("img.jpg")
canny_image=canny(image)
masked=region_of_interest(canny_image)
lines=cv.HoughLinesP(masked,2,np.pi/180,100,np.array([]),minLineLength=1,maxLineGap=5)
averaged_lines=average_slope_intercept(image,lines)
line_image=display_lines(image,averaged_lines)
combo_image=cv.addWeighted(image,0.8,line_image,1,1)
cv.imshow("result",combo_image)
cv.waitKey(0)

            
    
        
   