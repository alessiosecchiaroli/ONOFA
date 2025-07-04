import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from Pre_processing import standard_pre
from OF_solver import *
from Pyramidal_Horn_Schunck import HS_pyramidal
from scipy.ndimage.filters import convolve as filter2


# using the Pic_loader function to return the path
root = os.getcwd()
ref_img_path = os.path.join(root, './test images/sphere1.bmp')
work_img_path = os.path.join(root, './test images/sphere2.bmp')

ref_img = cv.imread(ref_img_path)
work_img = cv.imread(work_img_path)

ref_img_final = standard_pre(ref_img,1)
work_img_final = standard_pre(work_img,1)

u, v = HS_pyramidal(ref_img_final, work_img_final, alpha=25, levels=6, delta=1e-3,blr=1)
# u, v = lucas_kanade_optical_flow(ref_img_final, work_img_final, window_size=9)
# u, v, err =computeLK_tracking(ref_img_final, work_img_final)
draw_quiver(u,v,ref_img_final)
# draw_tracking(ref_img_final, u, v)

np.save("u_sphere_mine", u)
np.save("v_sphere_mine", v)


def get_derivatives(img1, img2):
    #derivative masks
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    fx = filter2(img1,x_kernel) + filter2(img2,x_kernel)
    fy = filter2(img1, y_kernel) + filter2(img2, y_kernel)
    ft = filter2(img1, -t_kernel) + filter2(img2, t_kernel)

    return [fx,fy, ft]



#input: images name, smoothing parameter, tolerance
#output: images variations (flow vectors u, v)
#calculates u,v vectors and draw quiver
def computeHS(name1, name2, alpha, delta):

    beforeImg = name1.astype(np.float64)
    afterImg = name2.astype(np.float64)
    #removing noise
    beforeImg  = cv.GaussianBlur(beforeImg, (5, 5), 0)
    afterImg = cv.GaussianBlur(afterImg, (5, 5), 0)

    # set up initial values
    u = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    v = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
    fx, fy, ft = get_derivatives(beforeImg, afterImg)
    avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                            [1 / 6, 0, 1 / 6],
                            [1 / 12, 1 / 6, 1 / 12]], float)
    iter_counter = 0
    while True:
        iter_counter += 1
        u_avg = filter2(u, avg_kernel)
        v_avg = filter2(v, avg_kernel)
        p = fx * u_avg + fy * v_avg + ft
        d = 4*alpha**2 + fx**2 + fy**2
        prev = u

        u = u_avg - fx * (p / d)
        v = v_avg - fy * (p / d)

        diff = np.linalg.norm(u - prev, 2)
        #converges check (at most 300 iterations)
        if  diff < delta or iter_counter > 300:
            # print("iteration number: ", iter_counter)
            break


    return [u, v]



u1, v1 = computeHS(ref_img_final, work_img_final, alpha=10, delta=1e-3)

draw_quiver(u1,v1,ref_img_final)
# draw_tracking(ref_img_final, u, v)

np.save("u_sphere_Git", u1)
np.save("v_sphere_Git", v1)