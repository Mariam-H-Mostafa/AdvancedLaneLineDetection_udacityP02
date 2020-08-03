import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

class calibrate_distort:

 def __init__(self,calpath):
    self.calpath = calpath
    

 def camera_cal(self,calpath):
# Read in Calibration Images
   images = glob.glob('C:\\Mariam\\Udacity-projects\\Udacity-P02\\P02\\AdvancedLanedetection\\AdvancedLanedetection\\camera_cal\\calibration*.jpg')

   #goal 1:Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

   # Define two arrays to store image points and object points from all images

   objpoints = [] # 3D points in real world space
   imgpoints = [] # 2D points in image plane

   objp = np.zeros((6*9,3),np.float32)
   objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

   for fname in images:
      img = mpimg.imread(fname)
      gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      #plt.imshow(gray)
      #plt.show()
      ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

      if ret == True:
         imgpoints.append(corners)
         objpoints.append(objp)
         #draw the corners
         img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
         #plt.imshow(img)
         #plt.show()
   

   ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
   
   return mtx, dist

#End of goal 1


#goal 2: Apply a distortion correction to raw images
 def undistortImg (self, img, mtx,dist):

    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


