# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 22:36:44 2021

@author: Emre Dogan
"""
import numpy as np
import pandas as pd
import cv2
import os
import glob
import matplotlib.pyplot as plt
import pickle



    # Prepare object points 0,0,0 ... 8,5,0
obj_pts = np.zeros((6*9,3), np.float32)
obj_pts[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Stores all object points & img points from all images
objpoints = []
imgpoints = []

    # Get directory for all calibration images
images = glob.glob('camera_cal/*.jpg')

for image in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    if ret == True:
            objpoints.append(obj_pts)
            imgpoints.append(corners)
    # Test undistortion on img
    img_size = (img.shape[1], img.shape[0])

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)
    def undistort(dst):
  
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    return dst
img = cv2.imread('camera_cal/calibration1.jpg')


dst = undistort(img)


    
    
    
    
    
    
    
    
    
    
    

