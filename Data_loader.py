import cv2 as cv
import os

def Pic_loader():
# find the picture
    root = os.getcwd()
    imgPath = os.path.join(root, '../Exp_pics/')

    return imgPath

# print(Pic_loader())