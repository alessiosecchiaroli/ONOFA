import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def eq_hist():
    root = os.getcwd ()
    imgPath = os.path.join (root, '../Exp_pics/220C_reference.bmp')
    img = cv.imread (imgPath)

    height, width, _ = img.shape
    scale = 1 / 4

    heightScale = int (height * scale)
    widthScale = int (width * scale)

    img = cv.resize (img, (widthScale, heightScale), interpolation=cv.INTER_LINEAR)
    img = cv.cvtColor (img, cv.COLOR_BGR2GRAY)

    dst = cv.equalizeHist(img)

    cv.imshow('img',img)
    cv.imshow('equalize img',dst)

    cv.waitKey(0)
#
# if __name__ == '__main__':
#     # cannyEdge()
#     openImage()

eq_hist()