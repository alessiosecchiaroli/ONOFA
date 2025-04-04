import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def callback(input):
    pass
def cannyEdge_visual(image):

    # blur if needed
    # image = cv.GaussianBlur(image,(3,3),3)

    height, width = image.shape
    scale = 1 / 5

    heightScale = int (height * scale)
    widthScale = int (width * scale)

    image = cv.resize (image, (widthScale, heightScale), interpolation=cv.INTER_LINEAR)

    winname = 'canny'
    cv.namedWindow(winname)
    cv.createTrackbar('minThres',winname,0,255,callback)
    cv.createTrackbar('maxThres',winname,0,255,callback)

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        minThres = cv.getTrackbarPos('minThres',winname)
        maxThres = cv.getTrackbarPos('maxThres',winname)
        cannyEdge = cv.Canny(image,minThres,maxThres)
        cv.imshow(winname,cannyEdge)

    cv.destroyAllWindows()