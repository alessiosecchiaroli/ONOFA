import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.ndimage import convolve

'''
The main algorithm and the masks have been taken from the Youtube video
Optical Flow: Horn and Schunck 
from Pratik Jain

The pyramidal implementation to improve the performance on larger displacements has been added by me (Alessio Secchiaroli).
Moreover, I added some blurring options to improve the convergence of the algorithm (i.e., reduce noise)

'''


def get_first_order_derivatives(img1, img2):
    #derivative masks
    #Opted Kernal convolution to efficiently implement Fourier transformations
    x_kernel = np.array([[-1, 1], [-1, 1]])
    y_kernel = np.array([[-1, -1], [1, 1]])
    t_kernel = np.ones((2, 2))

    fx = (convolve(img1, x_kernel) + convolve(img2, x_kernel))*0.5
    fy = (convolve(img1, y_kernel) + convolve(img2, y_kernel))*0.5
    ft = convolve(img1, -t_kernel) + convolve(img2, t_kernel)
    # ft = convolve(img1, t_kernel) + convolve(img2, -t_kernel)

    return [fx,fy,ft]


def HS_pyramidal(Image1,Image2, alpha, levels,delta=0.1,blr=5):

    Image1 = Image1.astype(np.float64) 
    Image2 = Image2.astype(np.float64) 

    Image1 = cv.GaussianBlur(Image1, (blr, blr), 0)
    Image2 = cv.GaussianBlur(Image2, (blr, blr), 0)

    # If using median blur, the float type is 32
    # Image1 = cv.medianBlur(Image1,blr)
    # Image2 = cv.medianBlur(Image2,blr)

    rows, cols = map(int, Image1.shape)

    for i in range(levels):

        Before_Img = Image1.copy()
        After_Img = Image2.copy()

        for _ in range(levels-1-i):
            Before_Img = cv.pyrDown(Before_Img) 
            After_Img = cv.pyrDown(After_Img)

        # set up initial values
        #2-D numpy array of zeros with the same shape as beforeImg

        if i == 0:
            u = np.zeros((Before_Img.shape[0], Before_Img.shape[1]))
            v = np.zeros((Before_Img.shape[0], Before_Img.shape[1]))
        else:
            u =  cv.pyrUp(u)    
            v =  cv.pyrUp(v)

            # Resize to match the current pyramid level's shape
            u = cv.resize(u, (Before_Img.shape[1], Before_Img.shape[0]), interpolation=cv.INTER_LINEAR)
            v = cv.resize(v, (Before_Img.shape[1], Before_Img.shape[0]), interpolation=cv.INTER_LINEAR)

        fx, fy, ft = get_first_order_derivatives(Before_Img, After_Img)
    
    
    # # The kernel with -1 as center element is the original Laplacian kernel suggested by Horn and Schunck in 1981
    
    # # The kernel with 0 as center element helps with the convergence of the algorithm by smoothing the flow field
    
        avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                                [1 / 6, -1, 1 / 6],
                                [1 / 12, 1 / 6, 1 / 12]], float)
        # avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
        #                         [1 / 6, 0, 1 / 6],
        #                         [1 / 12, 1 / 6, 1 / 12]], float)

        
        iter_counter = 0

        while True:
            iter_counter += 1
            u_avg = convolve(u, avg_kernel)
            v_avg = convolve(v, avg_kernel)


        #optical flow implementation
            p = (fx * u_avg) + (fy * v_avg) + ft 
            # # if using the original kernel, use this line
            # d = alpha**2 + fx**2 + fy**2
            # # if using the smoothing kernel, use this line instead      
            d = alpha + fx**2 + fy**2

            previous = u.copy()

            u = u_avg - fx * (p / d)
            v = v_avg - fy * (p / d)

            if not np.isfinite(u).all():
                print("Non-finite values in flow field — instability detected")
                break


            diff = np.linalg.norm(u - previous, 2)


            if  diff < delta:
                print("diff: ", diff)
                
                if i == levels:
                    break

                else:
                    # apply a 5x5 blur in intermediate steps (from Cakir paper)
                    u = u.astype(np.float32) # required for median blur
                    v = v.astype(np.float32) # required for median blur
                    u = cv.medianBlur(u,5)
                    v = cv.medianBlur(v,5)

                    break
            elif iter_counter > 10000:            
                # convergence error (at most 10000 iterations)
                print('diff:',diff)
                raise TypeError("Pyramidal level ",i ," failed to converge")



    return [u, v]