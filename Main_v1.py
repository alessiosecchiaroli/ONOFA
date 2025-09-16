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
from OF_solver import *
from Optical_Flow import draw_tracking
from Optical_Flow import computeLK_tracking
from Optical_Flow import lucas_kanade_optical_flow
from Pyramidal_Horn_Schunck import HS_pyramidal
from blob_detector_function import cross_finder
from cross_verification import match_score

# using the Pic_loader function to return the path
root = os.getcwd()
ref_img_path = os.path.join(root, './Exp_pics/BOS_8_5_1/BOS_8_5_10001.tif')
work_img_path = os.path.join(root, './Exp_pics/BOS_8_5_1/BOS_8_5_10002.tif')

ref_img = cv.imread(ref_img_path)
work_img = cv.imread(work_img_path)

# plt.subplot(1,2,1)
# plt.imshow(ref_img,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(work_img,cmap='gray')
# plt.show()

# standard pre-processing applied ( scale (1 means no scaling), and histogram equalization)
# ref_img = standard_pre(ref_img,1)
# work_img = standard_pre(work_img,1)

# no pre-processing, this step is required to get a single channel
ref_img = cv.cvtColor(ref_img, cv.COLOR_BGR2GRAY)
work_img = cv.cvtColor(work_img, cv.COLOR_BGR2GRAY)


# # subtract the images to see the difference
# trial = work_img - ref_img

# plt.imshow(trial)
# plt.show()

plt.subplot(1,2,1)
plt.imshow(ref_img,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(work_img,cmap='gray')    
plt.show()




# img1 = cv.GaussianBlur(ref_img,(5,5),0)
# img2 = cv.medianBlur(ref_img,5)

# plt.subplot(1,3,1)
# plt.imshow(ref_img,cmap='gray')
# plt.subplot(1,3,2)
# plt.imshow(img1,cmap='gray')
# plt.subplot(1,3,3)
# plt.imshow(img2,cmap='gray')    
# plt.show()

# Create a mask for the background region
# For example, assuming the background is a specific color or can be segmented
# Here, we create a dummy mask; replace this with your actual background mask
background_mask = np.ones (ref_img.shape[:2], dtype=bool)

# if it's first run at particular conditions, use this function to make a mask
# the script will brake after the mask is created, but a npy file will be created
# mask_point = mask_points(ref_img,"BOS_8_5_5_mask.npy")

# # otherwise use this one, adjust the name based on the npy file created
mask_point = np.load("Mask_shapes/BOS_8_5_1_mask.npy")
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

# ref_img_final = np.var(ref_img_final)
# work_img_final = np.var(ref_img_final)

cv.imwrite('Correlable_pics/BOS_8_5_1_masked.tif', work_img_final)
cv.imwrite('Correlable_pics/BOS_8_5_1_ref_masked.tif', ref_img_final)

plt.subplot(1,2,1)
plt.imshow(ref_img_final,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(work_img_final,cmap='gray')
plt.show()

# # ALPHA SENSITIVITY ANALYSIS
# alpha = np.linspace(1,100,10)
# blurs = [1, 3, 5, 7, 9, 11, 13, 15]

# for bb in blurs:
#     u, v = HS_pyramidal(ref_img_final, work_img_final, alpha=25, levels=6, delta=1e-3,blr=bb)
#     # draw_quiver(u,v,ref_img_final)

#     # Create filename strings with the current alpha value
#     alpha_str = f"{bb:.2f}"  # Format alpha to 2 decimal places
#     np.save(f"u-blur-{alpha_str}", u)
#     np.save(f"v-blur-{alpha_str}", v)

#     print('blur:', bb ,'completed')

u, v = HS_pyramidal(ref_img_final, work_img_final, alpha=25, levels=6, delta=1e-2, blr=5)
# u, v = lucas_kanade_optical_flow(ref_img_final, work_img_final, window_size=9)
# u, v, err =computeLK_tracking(ref_img_final, work_img_final)

draw_quiver(u,v,ref_img_final)
# # draw_tracking(ref_img_final, u, v)

np.save("u_HS", u)
np.save("v_HS", v)
# draw_OF_HS(ref_img_final, u, v, step = 10,scale = 1, color = 'red')
# print('debug')

# # If using Lucas-Kanade method for tracking the next lines plot the Optical flow 
# # Draw tracking overlays on the image
# tracked_img = draw_tracking(ref_img_final, p0, p1)

# # Convert image from BGR to RGB
# tracked_img_rgb = cv.cvtColor(tracked_img, cv.COLOR_BGR2RGB)

# # Plot using matplotlib
# plt.figure(figsize=(10, 6))
# plt.imshow(tracked_img_rgb)
# plt.title("Optical Flow Tracking")
# plt.axis("off")
# plt.show()





