import numpy as np
import matplotlib.pyplot as plt 
import cv2 as cv
from OF_solver import draw_quiver
from quick_plot import plot_midplane
import os
from scipy.signal import savgol_filter, medfilt

root = os.getcwd()
ref_img_path = os.path.join(root, './Correlable_pics/BOS_8_5_1_masked.tif')
ref_img = cv.imread(ref_img_path)

# plt.imshow(ref_img)
# plt.show()


u2= -1 * np.load("u_HS.npy")/25.4616
v2 = -1 * np.load("v_HS.npy")/25.4616


u = cv.GaussianBlur(u2,(5,5),0)
v2 = cv.GaussianBlur(v2,(5,5),0)


u3 = cv.GaussianBlur(u2,(7,7),0)
u4 = cv.GaussianBlur(u2,(11,11),0)
u5 = cv.GaussianBlur(u2,(33,33),0)
u6 = cv.GaussianBlur(u2,(51,51),0)
u7 = cv.GaussianBlur(u2,(111,111),0)

####

plot_midplane(u2,'raw')
plot_midplane(u,'5x5')
plot_midplane(u3,'7x7')
plot_midplane(u4,'11x11')
plot_midplane(u5,'33x33')
plot_midplane(u6,'51x51')
plot_midplane(u7,'111x111')





plt.legend()
plt.show()

# draw_quiver(u2,v2,ref_img)
