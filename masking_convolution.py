import numpy as np
from scipy.signal import convolve2d


def edge_detector(img):

    horizontal_kernel = np.array([[-0.25, 0, 0.25],
                                [-0.5, 0, 0.5],
                                [-0.25, 0, 0.25]])

    vertical_kernel = np.transpose(horizontal_kernel)

    # print(horizontal_kernel)
    # print(vertical_kernel)

    horiz_edge = convolve2d(img,horizontal_kernel)
    vert_edge = convolve2d(img,vertical_kernel)

    return horiz_edge, vert_edge


## PROVA ##

import matplotlib.pyplot as plt
import cv2 as cv

u = np.load('u_HS.npy') * -1

ho, ve = edge_detector(u)

plt.subplot(121)
plt.imshow(ho)
plt.subplot(122)
plt.imshow(ve)
plt.show()


