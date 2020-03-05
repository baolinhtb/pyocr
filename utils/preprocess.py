# written by junying, 2020-03-03

def showResult(name,
               img):
    cv2.imshow(name,img)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
import cv2

def colr2hsv(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    if 1:
        showResult("o",image)
        showResult("h",h)
        showResult("s",s)
        showResult("v",v)
    threshold(v)

import numpy as np
# module level variables ##########################################################################
GAUSSIAN_SMOOTH_FILTER_SIZE = (7, 7)
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
def maximizeContrast(imgGrayscale):

    height, width = imgGrayscale.shape

    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    if 0:
        showResult("imgTopHat",imgTopHat)
        showResult("imgBlackHat",imgBlackHat)
        showResult("imgGrayscalePlusTopHat",imgGrayscalePlusTopHat)
        showResult("imgGrayscalePlusTopHatMinusBlackHat",imgGrayscalePlusTopHatMinusBlackHat)

    return imgGrayscalePlusTopHatMinusBlackHat

def threshold(fspace):
    cv2.erode(fspace,(3,3),fspace)
    # Contrast
    contrasted = maximizeContrast(fspace)
    # Blur
    blurred = np.zeros(fspace.shape, np.uint8)
    blurred = cv2.GaussianBlur(fspace, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    # Threshold
    #_, thr = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thr = cv2.adaptiveThreshold(blurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT) 

    if 1:
        showResult("thr",thr)

if __name__ == "__main__":
    origin=cv2.imread('samples/2.jpg')
    h,w,c = origin.shape
    size = 200.0
    resized = cv2.resize(origin,(int(w*size/h),int(size)))
    colr2hsv(resized)