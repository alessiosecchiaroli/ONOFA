import numpy as np
import matplotlib.pyplot as plt 
import cv2 
from Pyramidal_Horn_Schunck import HS_pyramidal
from OF_plot import draw_OF_HS
import os
from video_maker import video_maker
from OF_plot import draw_quiver

'''
    This is a test script to validate the Pyramidal Horn-Schunck Optical Flow algorithm.
    It uses two images of a sphere taken at different times and computes the optical flow between them.
    The results are visualized using quiver plots.
    
    The images are taken from a Git-Hub repository and are proven to be valid.
'''

# name1 = 'sphere1.bmp'
# name2 = 'sphere2.bmp'	

# name1 = 'car1.jpg'
# name2 = 'car2.jpg'

name1 = 'table1.jpg'
name2 = 'table2.jpg'

path = os.path.join(os.path.dirname(__file__), 'test images')
beforeImg = cv2.imread(os.path.join(path, name1), cv2.IMREAD_GRAYSCALE)
afterImg = cv2.imread(os.path.join(path, name2), cv2.IMREAD_GRAYSCALE)

if beforeImg is None:
    raise NameError("Can't find image: \"" + name1 + '\"')
elif afterImg is None:
    raise NameError("Can't find image: \"" + name2 + '\"')

beforeImg = cv2.imread(os.path.join(path, name1), cv2.IMREAD_GRAYSCALE).astype(float)
afterImg = cv2.imread(os.path.join(path, name2), cv2.IMREAD_GRAYSCALE).astype(float)

# video_maker(beforeImg, afterImg)


u, v = HS_pyramidal(beforeImg, afterImg, alpha=0.1, levels=3, delta=0.01)
# draw_OF_HS(beforeImg, u, v, step = 10,scale = 1, color = 'red')
draw_quiver(u,v,beforeImg)
