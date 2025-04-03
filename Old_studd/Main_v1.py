import cv2 as cv
import os
import matplotlib.pyplot as plt
from bandwith_filter import bandpass_filter

# find the picture
root = os.getcwd()
imgPath = os.path.join(root, '../Exp_pics/220C_reference.bmp')
img = cv.imread(imgPath)

#rescale it
height, width, _ = img.shape
scale = 1/1

heightScale = int(height*scale)
widthScale = int(width*scale)

# equalize the histogram to improve contrast
img = cv.resize(img,(widthScale,heightScale),interpolation=cv.INTER_LINEAR)
img = cv.cvtColor (img, cv.COLOR_BGR2GRAY)
img = cv.equalizeHist (img)

plt.imshow(img,cmap='gray')
plt.show()

filtered = bandpass_filter(img,10,200)

plt.imshow(filtered,cmap='gray')
plt.show()

