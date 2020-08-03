import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from linetrack import line
class findlanes:
 
 def __init__(self,imgtransformed,Minv,origimg,Imgflag):
     self.binary_warped = imgtransformed
     self.Minv = Minv
     self.origimg = origimg
     self.Imgflag = Imgflag
     self.running_mean_difference_between_lines = 0
 
 def find_lane_pixels(self):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(self.binary_warped[self.binary_warped.shape[0]//2:,:], axis=0)
    print('hist',histogram)
    #plt.plot(histogram)
    #plt.show()
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # HYPERPARAMETERS
    
    nwindows = 9  # number of sliding windows  
    margin = 100  # The width of the windows +/- margin
    minpix = 50   # Set minimum number of pixels found to recenter window

    # Set height of windows - based on nwindows above and image shape  
    window_height = np.int(self.binary_warped.shape[0]//nwindows)
    nonzero = self.binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = self.binary_warped.shape[0] - (window+1)*window_height
        win_y_high = self.binary_warped.shape[0] - window*window_height
        win_xleft_low =  leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
   
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))        
                
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
       # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

   
    return leftx, lefty, rightx, righty, out_img

 def fit_poly(self,out_img, leftx, lefty, rightx, righty):

    img_shape = out_img.shape
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fitx, right_fitx, ploty,left_fit,right_fit

#############################################Video functions################################################3
 def get_averaged_line(previous_lines, new_line):
   
    # Number of frames to average over
    num_frames = 8
    
    if new_line is None:
        # No line was detected
        
        if len(previous_lines) == 0:
            # If there are no previous lines, return None
            return previous_lines, None
        else:
            # Else return the last line
            return previous_lines, previous_lines[-1]
    else:
        if len(previous_lines) < num_frames:
            # we need at least num_frames frames to average over
            previous_lines.append(new_line)
            return previous_lines, new_line
        else:
            # average over the last num_frames frames
            previous_lines[0:num_frames-1] = previous_lines[1:]
            previous_lines[num_frames-1] = new_line
            new_line = np.zeros_like(new_line)
            for i in range(num_frames):
                new_line += previous_lines[i]
            new_line /= num_frames
            return previous_lines, new_line

 def search_around_poly(self,leftline,rightline):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 60
    out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))

    # Grab activated pixels
    nonzero = self.binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (leftline.leftlinecoeff[0]*(nonzeroy**2) + leftline.leftlinecoeff[1]*nonzeroy + 
                    leftline.leftlinecoeff[2] - margin)) & (nonzerox < (leftline.leftlinecoeff[0]*(nonzeroy**2) + 
                    leftline.leftlinecoeff[1]*nonzeroy + leftline.leftlinecoeff[2] + margin)))
    right_lane_inds = ((nonzerox > (rightline.rightlinecoeff[0]*(nonzeroy**2) + rightline.rightlinecoeff[1]*nonzeroy + 
                    rightline.rightlinecoeff[2] - margin)) & (nonzerox < (rightline.rightlinecoeff[0]*(nonzeroy**2) + 
                    rightline.rightlinecoeff[1]*nonzeroy + rightline.rightlinecoeff[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if (leftx.size == 0 or rightx.size == 0):
        leftx, lefty, rightx, righty, out_img = findlanes.find_lane_pixels(self)    
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    
    # If no pixels were found return None
    ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    

    # Smoothing
    mean_difference = np.mean(right_fitx - left_fitx)
    left_curverad, right_curverad = findlanes.measure_curvature_real(self,ploty,left_fitx, right_fitx)
   
    if leftline.running_mean_difference_between_lines == 0:
       leftline.running_mean_difference_between_lines = mean_difference
        
    #if (mean_difference < 0.9*leftline.running_mean_difference_between_lines or mean_difference > 1.1*leftline.running_mean_difference_between_lines):
    #    if len(leftline.past_good_left_lines) == 0 and len(rightline.past_good_right_lines) == 0:
    #       leftx, lefty, rightx, righty, out_img = findlanes.find_lane_pixels(self)
    #       img_shape, left_fitx, right_fitx, ploty, left_fitC, right_fitC = findlanes.fit_poly(self,out_img, leftx, lefty, rightx, righty)
    #    else:
    #        left_fitx = leftline.past_good_left_lines[-1]
    #        right_fitx =rightline.past_good_right_lines[-1]
    if (left_curverad>1.1*leftline.left_curverad[-1] or left_curverad<0.8*leftline.left_curverad[-1]):
        leftx, lefty, rightx, righty, out_img= findlanes.find_lane_pixels(self)
        if len(rightx) == 0:
          print('it equals zero')

    if (right_curverad>1.1*rightline.right_curverad[-1] or right_curverad<0.8*rightline.right_curverad[-1]):
        leftx, lefty, rightx, righty, out_img = findlanes.find_lane_pixels(self)
        if len(rightx) == 0:
          print('it equals zero')
    else:
        leftline.past_good_left_lines, left_fitx = findlanes.get_averaged_line(leftline.past_good_left_lines, left_fitx)
        rightline.past_good_right_lines, right_fitx = findlanes.get_averaged_line(rightline.past_good_right_lines, right_fitx)
        mean_difference = np.mean(right_fitx - left_fitx)
        leftline.running_mean_difference_between_lines = 0.9*leftline.running_mean_difference_between_lines + 0.1*mean_difference
    
    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    if len(rightx) == 0:
        print('it equals zero')
    return out_img, left_fitx, right_fitx, ploty,left_fit,right_fit,leftx,rightx
   
 

 def drawing(self,left_fitx,right_fitx,ploty):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(self.binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, self.Minv, (self.origimg.shape[1], self.origimg.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(self.origimg, 1, newwarp, 0.3, 0)
    return result

 def measure_curvature_real(self,ploty, leftx, rightx):

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
   
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
   
    left_curverad = ((left_fit_cr[0]*y_eval*(2*ym_per_pix) + left_fit_cr[1])**2 + 1 )**(3/2) / np.absolute((2*left_fit_cr[0]))
    right_curverad = ((right_fit_cr[0]*y_eval*(2*ym_per_pix) + right_fit_cr[1])**2 + 1 )**(3/2) / np.absolute((2*right_fit_cr[0]))
    
    return left_curverad, right_curverad

 def car_position(self,leftx, rightx, xm_per_pix=3.7/800): 
    #car offset from the center
    ## Image mid horizontal position 
    mid_imgx = self.origimg.shape[1]//2
        
    ## Car position with respect to the lane
    car_pos = (leftx[-1] + rightx[-1])/2
    
    ## Horizontal car offset 
    car_pos = (mid_imgx - car_pos) * xm_per_pix

    return car_pos


 def find_drawlanes(self,n,leftline,rightline):
    if (self.Imgflag==True):
      leftx, lefty, rightx, righty, out_img = findlanes.find_lane_pixels(self)
      img_shape, left_fitx, right_fitx, ploty, left_fitC, right_fitC= findlanes.fit_poly(self,out_img, leftx, lefty, rightx, righty)
      #plt.imshow(out_img)
      #plt.show()
      result = findlanes.drawing(self,left_fitx,right_fitx,ploty)
      left_curverad, right_curverad = findlanes.measure_curvature_real(self,ploty,left_fitx, right_fitx)
      car_pos = findlanes.car_position(self,leftx, rightx, xm_per_pix=3.7/800)
   
    if (self.Imgflag==False): #In case of video
     if(n==0):
      leftx, lefty, rightx, righty, out_img = findlanes.find_lane_pixels(self)
      img_shape, left_fitx, right_fitx, ploty, left_fitC, right_fitC= findlanes.fit_poly(self,out_img, leftx, lefty, rightx, righty)
      #plt.imshow(out_img)
      #plt.show()
      result = findlanes.drawing(self,left_fitx,right_fitx,ploty)
      left_curverad, right_curverad = findlanes.measure_curvature_real(self,ploty,left_fitx, right_fitx)
      car_pos = findlanes.car_position(self,leftx, rightx, xm_per_pix=3.7/800)
      
      leftline.left_fitx = left_fitx
      leftline.left_curverad.append (left_curverad)
      leftline.leftlinecoeff = left_fitC
      
      rightline.right_curverad.append(right_curverad)
      rightline.right_fitx = right_fitx
      rightline.rightlinecoeff = right_fitC
      n=1
      
     else:
      print (n)
      out_img, left_fitx, right_fitx, ploty,left_fitC,right_fitC,leftx,rightx = findlanes.search_around_poly(self,leftline,rightline)
      
      result = findlanes.drawing(self,left_fitx,right_fitx,ploty)
      left_curverad, right_curverad = findlanes.measure_curvature_real(self,ploty,left_fitx, right_fitx)
      car_pos = findlanes.car_position(self,leftx, rightx, xm_per_pix=3.7/800)

      leftline.left_fitx = left_fitx
      leftline.left_curverad.append (left_curverad)
      leftline.leftlinecoeff = left_fitC
      
      rightline.right_curverad.append(right_curverad)
      rightline.right_fitx = right_fitx
      rightline.rightlinecoeff = right_fitC
    return result ,left_curverad, right_curverad,car_pos,n
