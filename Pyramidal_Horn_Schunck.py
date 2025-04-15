import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.ndimage import convolve

def get_first_order_derivatives(img1, img2):
    #derivative masks
    #Opted Kernal convolution to efficiently implement Fourier transformations
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    fx = convolve(img1,x_kernel) + convolve(img2,x_kernel)
    fy = convolve(img1, y_kernel) + convolve(img2, y_kernel)
    ft = convolve(img1, -t_kernel) + convolve(img2, t_kernel)

    return [fx,fy, ft]


def HS_pyramidal(Image1,Image2, alpha, levels,delta=0.1):

    Image1 = Image1.astype(np.float64) / 255.0
    Image2 = Image2.astype(np.float64) / 255.0
    # dividends = []
    # for j in range(levels):
    #     dividends.append(2**(levels-j))
    #     if j == levels:
    #         dividends.append(2)

    rows, cols = map(int, Image1.shape)

    for i in range(levels):

        Before_Img = Image1.copy()
        After_Img = Image2.copy()

        for _ in range(levels-1-i):
            Before_Img = cv.pyrDown(Before_Img) 
            After_Img = cv.pyrDown(After_Img)

        # Before_Img = cv.pyrDown(Image1,dstsize=(cols // dividends[i], rows // dividends[i]))    
        # After_Img = cv.pyrDown(Image2,dstsize=(cols // dividends[i], rows // dividends[i]))

        # set up initial values
        #2-D numpy array of zeros with the same shape as beforeImg

        if i == 0:
            u = np.zeros((Before_Img.shape[0], Before_Img.shape[1]))
            v = np.zeros((Before_Img.shape[0], Before_Img.shape[1]))
        else:
            u = cv.pyrUp(u)    
            v = cv.pyrUp(v)

            # Resize to match the current pyramid level's shape
            u = cv.resize(u, (Before_Img.shape[1], Before_Img.shape[0]), interpolation=cv.INTER_LINEAR)
            v = cv.resize(v, (Before_Img.shape[1], Before_Img.shape[0]), interpolation=cv.INTER_LINEAR)

        # u = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
        # v = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
        fx, fy, ft = get_first_order_derivatives(Before_Img, After_Img)
    
        avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                                [1 / 6, 0, 1 / 6],
                                [1 / 12, 1 / 6, 1 / 12]], float)
        iter_counter = 0
    
        while True:
            iter_counter += 1
            u_avg = convolve(u, avg_kernel)
            v_avg = convolve(v, avg_kernel)


        #optical flow implementation
            p = (fx * u_avg) + (fy * v_avg) + ft       
            d = 4 * alpha**2 + fx**2 + fy**2

            previous = u.copy()

            u = u_avg - fx * (p / d)
            v = v_avg - fy * (p / d)

            if not np.isfinite(u).all():
                print("Non-finite values in flow field — instability detected")
                break


            diff = np.linalg.norm(u - previous, 2)
            #converges check (at most 300 iterations)
            if  diff < delta or iter_counter > 300:
                # print("iteration number: ", iter_counter)
                break


    return [u, v]