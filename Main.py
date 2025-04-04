import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

from Filters import *
from Data_loader import Pic_loader
from Pre_processing import standard_pre
from Canny_visualizer import cannyEdge_visual

# using the Pic_loader function to return the path
root = os.getcwd()
imgPath = os.path.join(root, './Exp_pics/220C_reference.bmp')
img = cv.imread(imgPath)

# plt.imshow(img)
# plt.show()

# standard pre-processing applied
img = standard_pre(img,1)

# plotting for debugging
# plt.imshow(img, cmap='gray')
# plt.show()


band_filtered_image = bandpass_filter(img,20,180)
#
# # #plotting for debugging
# plt.imshow(band_filtered_image,cmap='gray')
# plt.show()

# img_sub = Blur_subtraction(img,99)

# plt.imshow(img_sub,cmap='gray')
# plt.show()

# canny edge visualizer is used to tune the values
# thanks to the trackbar, it is easy to visualize the effect of the limiting values

# cannyEdge_visual(img_sub)

# min and max treshold are based on the visual version of Canny edge
# edges = cv.Canny(band_filtered_image,100,255)
#
# plt.imshow(edges)
# plt.show()

circles = cv.HoughCircles(band_filtered_image,cv.HOUGH_GRADIENT,1,200,param1=255,param2=100,minRadius=40,maxRadius=160)
circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    # draw the outer circle
    cv.circle (img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv.circle (img, (i[0], i[1]), 2, (0, 0, 255), 3)

cv.imshow ('detected circles', img)
cv.waitKey (0)
cv.destroyAllWindows ()