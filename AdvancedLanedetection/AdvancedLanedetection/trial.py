import cv2

# Opens the Video file
cap= cv2.VideoCapture("C:\\Mariam\\Udacity-projects\\Udacity-P02\\CarND-Advanced-Lane-Lines-master\\CarND-Advanced-Lane-Lines-master\\project_video.mp4")
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite("C:\\Mariam\\Udacity-projects\\Udacity-P02\\CarND-Advanced-Lane-Lines-master\\CarND-Advanced-Lane-Lines-master\\video_images"+str(i)+'.jpg',frame)
    i+=1

cap.release()
cv2.destroyAllWindows()
