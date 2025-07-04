import numpy as np
import matplotlib.pyplot as plt 
import cv2 as cv
from OF_solver import draw_quiver
from quick_plot import plot_midplane
import os

root = os.getcwd()
ref_img_path = os.path.join(root, './Correlable_pics/220C_ref_masked.bmp')
ref_img = cv.imread(ref_img_path)

# plt.imshow(ref_img)
# plt.show()

u= np.load("u_HS.npy")
v = np.load("v_HS.npy")
u1 = np.load("u_LK.npy")
v1 = np.load("v_LK.npy")

# u = np.load("u_sphere_mine.npy")
# v = np.load("v_sphere_mine.npy")
# us = np.load("u_sphere_Git.npy")
# vs = np.load("v_sphere_Git.npy")

plot_midplane(u, 'u mine')
# plot_midplane(v, 'v mine')
plot_midplane(u1, 'u LK')
# plot_midplane(v1, 'v LK')
# plot_midplane(us, 'u Git')
# plot_midplane(vs, 'v Git')
plt.legend()
plt.show()

# u0 = np.load("u-blur-1.00.npy")
# u3 = np.load("u-blur-3.00.npy")
# u5 = np.load("u-blur-5.00.npy")
# u7 = np.load("u-blur-7.00.npy")
# u9 = np.load("u-blur-9.00.npy")
# u11 = np.load("u-blur-11.00.npy")
# u13 = np.load("u-blur-13.00.npy")
# u15 = np.load("u-blur-15.00.npy")

# plot_midplane(-1* u0, 'Blur 1')
# plot_midplane(-1 * u3, 'Blur 3')
# plot_midplane(-1* u5, 'Blur 5')
# plot_midplane(-1* u7, 'Blur 7')
# plot_midplane(-1* u9, 'Blur 9')
# plot_midplane(-1* u11, 'Blur 11')
# plot_midplane(-1* u13, 'Blur 13')
# plot_midplane(-1* u15, 'Blur 15')
# plt.legend()
# plt.show()
