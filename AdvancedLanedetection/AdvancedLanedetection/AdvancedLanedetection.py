import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from calibrate_distort import calibrate_distort
from imgvidprocessing import imgvidprocessing
from findlanes import findlanes
from linetrack import line

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2


Imgflag = False

n = 0
cal_path = 'C:\\Mariam\\Udacity-projects\\Udacity-P02\\P02\\AdvancedLanedetection\\AdvancedLanedetection\\camera_cal\\calibration*.jpg'

calibrat_and_undistort = calibrate_distort(cal_path)
mtx,dist = calibrat_and_undistort.camera_cal(cal_path)
leftline = line()   
rightline = line()


def pipline(img):
    global n
    undistortedimg = calibrat_and_undistort.undistortImg (img, mtx,dist)
    if Imgflag == True:
     cv2.imwrite("C:\\Mariam\\Udacity-projects\\Udacity-P02\\CarND-Advanced-Lane-Lines-master\\CarND-Advanced-Lane-Lines-master\\output_images\\"+testimgname+'_undist'+'.jpg', undistortedimg)

    s_thresh = (100,255)
    sx_thresh = (20,100)
    sobel_kernel = 3

    pimgv = imgvidprocessing(undistortedimg, s_thresh, sx_thresh, sobel_kernel)

    warpedImgtransformed,Minv = pimgv.gen_binaryimg()
    if Imgflag == True:
     cv2.imwrite("C:\\Mariam\\Udacity-projects\\Udacity-P02\\CarND-Advanced-Lane-Lines-master\\CarND-Advanced-Lane-Lines-master\\output_images\\"+testimgname+'_warped'+'.jpg', warpedImgtransformed)

    findlanes_drawing_curvature = findlanes(warpedImgtransformed,Minv,img,Imgflag)

    finalimg ,left_curverad, right_curverad, car_pos, n_current = findlanes_drawing_curvature.find_drawlanes(n,leftline,rightline)   
    n=n_current
    if Imgflag == True:
       cv2.putText(finalimg,'left curve' + str(left_curverad), (10,500), font, fontScale,fontColor,lineType)
       cv2.putText(finalimg,'right curve' + str(right_curverad), (10,540), font, fontScale,fontColor,lineType)
       cv2.putText(finalimg,'car position'+ str(car_pos), (10,580), font, fontScale,fontColor,lineType)
       cv2.imwrite("C:\\Mariam\\Udacity-projects\\Udacity-P02\\CarND-Advanced-Lane-Lines-master\\CarND-Advanced-Lane-Lines-master\\output_images\\"+testimgname+'_final'+'.jpg', finalimg)
    if Imgflag == False:
        leftCurve = sum(leftline.left_curverad)/len(leftline.left_curverad)
        rightCurve = sum(rightline.right_curverad)/len(rightline.right_curverad)
        cv2.putText(finalimg,'left curve' + str(leftCurve), (10,500), font, fontScale,fontColor,lineType)
        cv2.putText(finalimg,'right curve' + str(rightCurve), (10,540), font, fontScale,fontColor,lineType)
        cv2.putText(finalimg,'car position'+ str(car_pos), (10,580), font, fontScale,fontColor,lineType)

    return finalimg 

if Imgflag==True:
   distimgpath = 'C:\\Mariam\\Udacity-projects\\Udacity-P02\\CarND-Advanced-Lane-Lines-master\\CarND-Advanced-Lane-Lines-master\\sample'
   testimages = os.listdir(distimgpath)
   for testimg in testimages:
       img = mpimg.imread('C:\\Mariam\\Udacity-projects\\Udacity-P02\\CarND-Advanced-Lane-Lines-master\\CarND-Advanced-Lane-Lines-master\\sample\\'+testimg)
       testimgname = testimg.split('.jpg')
       testimgname = testimgname[0]
       finalimg  = pipline(img)

if Imgflag == False:
    video_output_path = "C:\\Mariam\\Udacity-projects\\Udacity-P02\\CarND-Advanced-Lane-Lines-master\\CarND-Advanced-Lane-Lines-master\\project_video_processed.mp4"
    video_input = VideoFileClip("C:\\Mariam\\Udacity-projects\\Udacity-P02\\CarND-Advanced-Lane-Lines-master\\CarND-Advanced-Lane-Lines-master\\project_video.mp4")
    processed_video = video_input.fl_image(pipline)    
    processed_video.write_videofile(video_output_path, audio=False)

