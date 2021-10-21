import cv2 as cv
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import argparse
import imutils

image = cv.imread(r'D:\3D-Object-Detection\Photos\coins.jpg')
width,height = 700,500
image = cv.resize(image,(width,height))
shifted = cv.pyrMeanShiftFiltering(image,100,100)

gray = cv.cvtColor(shifted,cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

cnts = cv.findContours(thresh.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("[INFO] {} in unique contours found".format(len(cnts)))

for (i,c) in enumerate(cnts):
    ((x,y),_) = cv.minEnclosingCircle(c)
    # cv.putText(image,"#{}".format(i+1),(int(x)-1,int(y)),cv.FONT_HERSHEY_SCRIPT_SIMPLEX,0.6,(0,0,255),2)
    cv.drawContours(image,[c],-1,(0,255,0),2)

D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D,indices = False,min_distance=10,labels=thresh)

markers = ndimage.label(localMax,structure=np.ones((3,3)))[0]
labels = watershed(-D,markers,mask=thresh)

for label in np.unique(labels):
    if label == 0:
        continue
    mask = np.zeros(gray.shape,dtype="uint8")
    mask[labels==label] = 255

    cnts = cv.findContours(mask.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts,key=cv.contourArea)

    ((x,y),r) = cv.minEnclosingCircle(c)
    cv.circle(image,(int(x),int(y)),int(r),(0,255,0),2)

cv.imshow("Input",image)
cv.waitKey(0)


