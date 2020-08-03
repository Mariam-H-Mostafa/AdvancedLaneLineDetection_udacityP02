import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
class imgvidprocessing:

 def __init__(self,undistortedimg,s_thresh, sx_thresh,sobel_kernel):
     self.s_thresh = s_thresh
     self.sx_thresh = sx_thresh
     self.sobel_kernel = sobel_kernel
     self.img = undistortedimg

 def abs_sobel_thresh(self,orient = 'x'):
    # Calculate directional gradient
    #Note: img is the undistorted image
    # Apply threshold
    gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    grad_binary = np.zeros_like(scaled_sobel) 
    grad_binary[(scaled_sobel > self.sx_thresh[0]) & (scaled_sobel < self.sx_thresh[1])] = 1
    #plt.imshow(grad_binary, cmap='gray')
   # plt.show()
    return grad_binary

 def mag_thresh(self, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    #Note: img is the undistorted image
    # Apply threshold
    gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0,sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1,sobel_kernel)


    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    mag_binary = np.zeros_like(gradmag)
    
    mag_binary[(gradmag > mag_thresh[0]) & (gradmag < mag_thresh[1])] = 1
    return mag_binary

 def dir_threshold(self,thresh=(0, np.pi/2)):
    # Calculate gradient direction
    #Note: img is the undistorted image
    # Apply threshold
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(self.img, cv2.CV_64F, 1, 0,ksize=self.sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1,ksize=self.sobel_kernel)
    
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    direction_gradient = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(direction_gradient)
    
    dir_binary[(direction_gradient >= thresh[0]) & (direction_gradient <= thresh[1])] = 1

    return dir_binary

 def hls_select(self):
    #Note: img is the undistorted image
  
    hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    
    Sbinary = np.zeros_like(S)
    Sbinary[(S > self.s_thresh[0]) & (S <= self.s_thresh[1])] = 1
    #plt.imshow(Sbinary)
    #plt.show()
    return Sbinary

 def perspectivetransform (self,binaryimg):
    #img here is the undistorted image
    img_width  = binaryimg.shape[1]
    img_length = binaryimg.shape[0]
    mid_width = 0.8
    offset = 50
    img_size = (binaryimg.shape[1], binaryimg.shape[0])
    src_points = np.float32([[(img_width/2)-offset,img_length*.6],[(img_width/2)+offset,img_length*.6],[img_width-((img_width)/3.5),img_length*.95],[(img_width)/3.5,img_length*.95]])
    dst_points = np.float32([[(img_width)/3.5,0],[img_width-((img_width)/3.5),0],[img_width-((img_width)/3.5),img_length],[(img_width)/3.5,img_length]])


    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped = cv2.warpPerspective(binaryimg, M, img_size, flags=cv2.INTER_LINEAR)
    #plt.imshow(warped,cmap='gray')
    #plt.show()
    return warped,Minv

 def gen_binaryimg(self):
    #Note: img is the undistorted image
    Sbinary = imgvidprocessing.hls_select(self)

    grad_binary = imgvidprocessing.abs_sobel_thresh(self, orient = 'x')

    combined_binary = np.zeros_like(grad_binary)
    combined_binary[(Sbinary == 1) | (grad_binary == 1)] = 1
    
    warpedImgtransformed,Minv = imgvidprocessing.perspectivetransform (self,combined_binary)
    return  warpedImgtransformed*255,Minv
