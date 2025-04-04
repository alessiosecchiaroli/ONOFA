import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# This function pre-process the image in a standardize way
# Scale it, if needed
# Transform RGB to gray scale
# Equalize the histogram for better contrast
# plot the processed image (if needed)
def standard_pre(img,scale_factor):
    # rescale it
    height, width, _ = img.shape
    scale = 1 / scale_factor

    heightScale = int (height * scale)
    widthScale = int (width * scale)

    # equalize the histogram to improve contrast
    img = cv.resize (img, (widthScale, heightScale), interpolation=cv.INTER_LINEAR)
    img = cv.cvtColor (img, cv.COLOR_BGR2GRAY)
    image_processed = cv.equalizeHist (img)

    # plotting for debugging, it works :)
    # plt.imshow (image_processed, cmap='gray')
    # plt.show()

    return image_processed