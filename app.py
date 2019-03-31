#link opencv-contrib-python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(lane_image):
    gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

def region_interest(image):
    #shape m,n,l: x y z
    height = image.shape[0]
    polygon = np.array([ [(200,height),(1100,height),(550,250)] ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygon,255)
    
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            print(line)
            #convert from [[x1,y1,x2,y2]] -> [x1,y1,x2,y2]
            x1,y1,x2,y2 = line.reshape(4)
            #cv2.line(image,p1,p2,color,line width)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image

image = cv2.imread('test_image.jpeg')
lane_image = np.copy(image)

canny = canny(lane_image)
cropped_img = region_interest(canny)

#hough lines (image,grid size,precision,no of votes for cell,empty array,min Line,max Gap)
lines = cv2.HoughLinesP(cropped_img, 2, np.pi/180, 100, np.array([]), minLineLength = 40,maxLineGap = 5)

#draw the found lines on original image
line_image = display_lines(lane_image,lines)
#addWeighted(img1*0.8,img2*1,scalar)
combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)

cv2.imshow('Result',combo_image)
cv2.waitKey(0)
