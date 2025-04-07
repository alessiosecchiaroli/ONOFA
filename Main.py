import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

from Filters import *
from Data_loader import Pic_loader
from Pre_processing import standard_pre
from Canny_visualizer import cannyEdge_visual
from circle_finder import circles_finder
from video_maker import video_maker
from Masking import mask_points
from Masking import shape_isolation
from OF_solver import computeHS

# using the Pic_loader function to return the path
root = os.getcwd()
ref_img_path = os.path.join(root, './Exp_pics/220C_reference.bmp')
work_img_path = os.path.join(root, './Exp_pics/220C_4bar.bmp')

ref_img = cv.imread(ref_img_path)
work_img = cv.imread(work_img_path)

# plt.imshow(img)
# plt.show()

# standard pre-processing applied
ref_img = standard_pre(ref_img,1)
work_img = standard_pre(work_img,1)

# plt.subplot(1,2,1)
# plt.imshow(ref_img)
# plt.subplot(1,2,2)
# plt.imshow(work_img)
# plt.show()

crosses_reference = np.array(circles_finder(ref_img,11, 850, 900))
crosses_work = np.array(circles_finder(work_img,11, 860, 910))

# print(crosses_reference)
# print(crosses_work)

difference = crosses_reference.astype(np.int16) - crosses_work.astype(np.int16)

if np.all(difference[0, :] == difference[1, :]):
    print("Rows are identical: good :)")
else:
    print("Rows are different: there might be an error :(")


work_img = work_img[4:,:]
ref_img = ref_img[:-4,:]

# cv.imwrite('220C_4bar_corrected.bmp', work_img)
# cv.imwrite('220C_ref_corrected.bmp', ref_img)

# if it's first run at particular conditions, use this function to make a mask
# mask_points = mask_points(ref_img)

# otherwise use this one, adjust the name
mask_points = np.load("Mask_shapes/220C_mask_points.npy")
mask_points = mask_points.reshape(63,2)


# mask = cv.polylines (ref_img, [mask_points], isClosed=True, color=(0, 0, 0), thickness=3)
#
# ref_img_M = cv.bitwise_and (ref_img, ref_img, mask=mask)
# work_img_M = cv.bitwise_and (work_img, work_img, mask=mask)

ref_img_final = shape_isolation(ref_img,mask_points)
work_img_final = shape_isolation(work_img,mask_points)

# cv.imwrite('Correlable_pics/220C_4bar_masked.bmp', work_img_final)
# cv.imwrite('Correlable_pics/220C_ref_masked.bmp', ref_img_final)

# plt.subplot(1,2,1)
# plt.imshow(ref_img_final)
# plt.subplot(1,2,2)
# plt.imshow(work_img_final)
# plt.show()

# make a video, required for OF
# BOS_220C_4bar = video_maker(ref_img_final,work_img_final)


u,v = computeHS(ref_img_final, work_img_final, alpha = 15, delta = 1)





