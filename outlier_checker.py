import numpy as np
from OF_plot import draw_quiver
from quick_plot import plot_midplane
from outlier import normres
import cv2 as cv 
import matplotlib.pyplot as plt

ref_img_final= cv.imread('Correlable_pics/BOS_8_6_3_ref_masked.tif')

u = np.load('u_HS.npy')
v = np.load('v_HS.npy')

Info, u_new, v_new = normres(u,v,0.1)

plot_midplane(-1* u,'no filter')
plot_midplane(-1* u_new, 'filtered')

plt.legend()
plt.show()

# draw_quiver(u_new,v_new,ref_img_final)