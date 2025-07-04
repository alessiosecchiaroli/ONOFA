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
ref_img_path = os.path.join(root, './Exp_pics/220C_ref_prealigned.tif')
work_img_path = os.path.join(root, './Exp_pics/220C_9bar_prealigned.tif')

ref_img = cv.imread(ref_img_path)
work_img = cv.imread(work_img_path)

# plt.subplot(1,2,1)
# plt.imshow(ref_img,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(work_img,cmap='gray')
# plt.show()

# standard pre-processing applied ( scale (1 means no scaling), and histogram equalization)
ref_img = standard_pre(ref_img,1)
work_img = standard_pre(work_img,1)

# # subtract the images to see the difference
# trial = work_img - ref_img

# plt.imshow(trial)
# plt.show()

# plt.subplot(1,2,1)
# plt.imshow(ref_img,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(work_img,cmap='gray')    
# plt.show()


# # # circle finder, use a circle hough transform to find the circles
# # # it used to work with the old pictures, but now although the quality is better, the circles are not detected
# # # I DON'T KNOW WHY	 :( :( :(
# crosses_reference = np.array(circles_finder(ref_img,11, 830, 910,48,4))
# crosses_work = np.array(circles_finder(work_img,11, 830, 910,48,4))

# # NEW ERA --> BLOB DETECTOR
# # it's still somehow hardcoded, I have to input two different level of blurriness to find the top and bottom crosses
# # T = TOP, B = BOTTOM

# # Cross Reference 
# CRT_pos, CRT_size = cross_finder(ref_img,13) # TOP
# CRB_pos, CRB_size = cross_finder(ref_img,17) # BOTTOM

# CRT = np.append(CRT_pos, CRT_size)
# CRB = np.append(CRB_pos, CRB_size)

# crosses_reference = np.vstack((CRT, CRB))

# # # Cross Working
# CWT_pos, CWT_size = cross_finder(work_img,13)
# CWB_pos, CWB_size = cross_finder(work_img,17)

# CWT = np.append(CWT_pos, CWT_size)
# CWB = np.append(CWB_pos, CWB_size)

# crosses_work = np.vstack((CWT, CWB))

# # crosses_reference = np.append(crosses_reference_top, crosses_reference_bottom, axis=0)
# print(crosses_reference)
# print(crosses_work)

# conditions, score = match_score(crosses_reference, crosses_work)

# for cond, result in conditions.items():
#     print(f"{cond}: {'✔' if result else '✘'}")

# # print(f"Total match score: {score}/4")

# ref_img = ref_img[:-4,:-1]
# work_img = work_img[4:,1:]


# subtract the images to see the difference
# trial = work_img - ref_img

# plt.imshow(trial)
# plt.show()

# Create a mask for the background region
# For example, assuming the background is a specific color or can be segmented
# Here, we create a dummy mask; replace this with your actual background mask
background_mask = np.ones (ref_img.shape[:2], dtype=bool)

# Adjust the target image to match the reference image's background intensity
# adjusted_image = match_background_intensity_gray (ref_img, work_img, background_mask)
# BOS_220C_9bar = video_maker(ref_img,work_img)

# cv.imwrite('220C_4bar_corrected.bmp', work_img)
# cv.imwrite('220C_ref_corrected.bmp', ref_img)

# if it's first run at particular conditions, use this function to make a mask
# the script will brake after the mask is created, but a npy file will be created
# mask_point = mask_points(ref_img)

# # otherwise use this one, adjust the name based on the npy file created
mask_point = np.load("Mask_shapes/220C_prealigned.npy")
mask_point = mask_point.reshape(77,2)


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

cv.imwrite('Correlable_pics/220C_9bar_masked.bmp', work_img_final)
cv.imwrite('Correlable_pics/220C_ref_masked.bmp', ref_img_final)

# plt.subplot(1,2,1)
# plt.imshow(ref_img_final)
# plt.subplot(1,2,2)
# plt.imshow(work_img_final)
# plt.show()

# make a video, required for OF
# BOS_220C_9bar = video_maker(ref_img_final,work_img_final)

# ref_img_final = cv.normalize(ref_img_final,ref_img_final,alpha = 255, beta=0, norm_type=cv.NORM_MINMAX)
# ref_img_final = cv.normalize(ref_img_final,ref_img_final,0,255, norm_type=cv.NORM_MINMAX)
# work_img_final = cv.normalize(work_img_final,work_img_final,0, 255, norm_type=cv.NORM_MINMAX)
# work_img_final = cv.normalize(work_img_final,work_img_final,alpha = 255, beta=0, norm_type=cv.NORM_MINMAX)

# check = ref_img_final-work_img_final

# plt.imshow(check)
# plt.show()

# ref_img_final = Blur_subtraction(ref_img_final,99)
# work_img_final = Blur_subtraction(work_img_final,99)

# plt.subplot(1,2,1)
# plt.imshow(ref_img_final,cmap='gray')
# plt.subplot(1,2,2)
# plt.imshow(work_img_final,cmap='gray')
# plt.show()

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

# u, v = HS_pyramidal(ref_img_final, work_img_final, alpha=25, levels=6, delta=1e-3,blr=5)
u, v = lucas_kanade_optical_flow(ref_img_final, work_img_final, window_size=9)
# u, v, err =computeLK_tracking(ref_img_final, work_img_final)
draw_quiver(u,v,ref_img_final)
# # draw_tracking(ref_img_final, u, v)

np.save("u_LK", u)
np.save("v_LK", v)
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





