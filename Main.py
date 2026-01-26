import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
# from Filters import *
from Pre_processing import standard_pre
# from Canny_visualizer import cannyEdge_visual
# from circle_finder import circles_finder
from video_maker import video_maker
from Masking import mask_points
from Masking import shape_isolation
from OF_plot import *
from Pyramidal_Horn_Schunck_tqdm import HS_pyramidal
# from blob_detector_function import cross_finder
# from cross_verification import match_score

# using the Pic_loader function to return the path
root = os.getcwd()
ref_img_path = os.path.join(root, './Exp_pics/BOS_12_11/BOS_220C_reference0001.tif')
work_img_path = os.path.join(root, './Exp_pics/BOS_12_11/BOS_12_11_10001.tif')

ref_img = cv.imread(ref_img_path)
work_img = cv.imread(work_img_path)

# # Visualize the raw images

# plt.subplot(1,2,1)
# plt.imshow(ref_img,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(work_img,cmap='gray')
# plt.show()

# # Standard pre-processing applied ( scale (1 means no scaling), and histogram equalization)
ref_img = standard_pre(ref_img,1)
work_img = standard_pre(work_img,1)

# # no pre-processing, this step is required to get a single channel
# # Only for other file formats than tiff

# ref_img = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
# work_img = cv.cvtColor(work_img, cv.COLOR_BGR2GRAY)

# # Visualize the normalized images

plt.subplot(1,2,1)
plt.imshow(ref_img,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(work_img,cmap='gray')    
plt.show()

'''
Quick check for visualizing the displacement between the two images
'''
# # subtract the images to see the difference
# trial = work_img - ref_img

# plt.imshow(trial)
# plt.show()

# Create a mask for the background region
# For example, assuming the background is a specific color or can be segmented
# Here, a dummy mask is created; replace this with your actual background mask
background_mask = np.ones (ref_img.shape[:2], dtype=bool)

# # If it's first run at particular conditions (i.e., BOS_x_y_z), use this function to create a mask
# # The script will brake after the mask is created, but a npy file will be created

# mask_point = mask_points(ref_img,"BOS_12_11_1_mask.npy")

# # If a mask already exists, use this line, adjust the name based on the npy file created
mask_point = np.load("Mask_shapes/BOS_12_11_1_mask.npy")
mask_len = np.size(mask_point)
mask_point = mask_point.reshape(int(mask_len/2),2)

mask = cv.polylines (ref_img, [mask_point], isClosed=True, color=(0, 0, 0), thickness=3)

ref_img_M = cv.bitwise_and (ref_img, ref_img, mask=mask)
work_img_M = cv.bitwise_and (work_img, work_img, mask=mask)

ref_img_final = shape_isolation(ref_img,mask_point)
work_img_final = shape_isolation(work_img,mask_point)

# find the min and max y, to reduce the frames' dimensions
max_y = np.max(mask_point[:,1])
min_y = np.min(mask_point[:,1])

# slice the picture
ref_img_final = ref_img_final[min_y:max_y,:]
work_img_final = work_img_final[min_y:max_y,:]

# Save the masked images 
cv.imwrite('Correlable_pics/BOS_12_11_1_masked.tif', work_img_final)
cv.imwrite('Correlable_pics/BOS_12_11_ref_masked.tif', ref_img_final)

# visualize the masked images
plt.subplot(1,2,1)
plt.imshow(ref_img_final,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(work_img_final,cmap='gray')
plt.show()

# # MAIN RUN
# # The number of levels is determined based on the maximum displacement expected
# # I keep 6 levels based on literature: https://doi.org/10.1007/s00348-022-03553-z 
# # The blur is based on the results from my Cross-Correlation pre-processubg
# # Alpha is based on some trial and error
u, v = HS_pyramidal(ref_img_final, work_img_final, alpha=25, levels=6, delta=1e-2, blr=5)

# # Visualize the results
draw_quiver(u,v,ref_img_final)

# Save the results
np.save("u_HS", u)
np.save("v_HS", v)
