import numpy as np
import cv2 as cv
from scipy.ndimage.filters import convolve 
import matplotlib.pyplot as plt
from scipy.linalg import norm 

def get_derivatives(img1, img2):
    #derivative masks
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    fx = convolve(img1, x_kernel) + convolve(img2, x_kernel)
    fy = convolve(img1, y_kernel) + convolve(img2, y_kernel)
    ft = convolve(img1, -t_kernel) + convolve(img2, t_kernel)

    return [fx,fy, ft]

# Tried to downsample the image by using a Gaussian filter, wrong!! 
def computeHS(Image1, Image2, alpha, delta, levels):

    # use Pyramid levels to gradually solve OF
    # The size 19 px to 3 px are hardcoded
    # All odd integers from 19 to 3
    odds = np.arange(19, 2, -2)

    # Now select `levels` number of values evenly spaced from that list
    pyramid_lvl = np.linspace(0, len(odds) - 1, levels, dtype=int)
    pyramid_lvl = odds[pyramid_lvl]

    for i in range(levels):
        #removing noise
        beforeImg  = cv.GaussianBlur(Image1, (pyramid_lvl[i],pyramid_lvl[i]),0)
        afterImg = cv.GaussianBlur(Image2, (pyramid_lvl[i],pyramid_lvl[i]),0)

        if i == 0:
            # set up initial values
            u = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
            v = np.zeros((beforeImg.shape[0], beforeImg.shape[1]))
        else:
            u=u
            v=v

        fx, fy, ft = get_derivatives(beforeImg, afterImg)
        avg_kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                                [1 / 6, 0, 1 / 6],
                                [1 / 12, 1 / 6, 1 / 12]], float)
        iter_counter = 0
        while True:
            iter_counter += 1
            u_avg = convolve(u, avg_kernel)
            v_avg = convolve(v, avg_kernel)
            p = fx * u_avg + fy * v_avg + ft
            d = 4 * alpha**2 + fx**2 + fy**2
            prev = u

            u = u_avg - fx * (p / d)
            v = v_avg - fy * (p / d)

            diff = np.linalg.norm(u - prev, 2)
            # diff = np.linalg.norm(u - prev)
            # diff = norm(u-prev)
            #converges check (at most 300 iterations)
            if  diff < delta or iter_counter > 1000:
                # print("iteration number: ", iter_counter)
                break

    return [u, v]


# Assume Image is the grayscale image, u and v are the optical flow fields
def draw_OF_HS(Image, u, v, step = 10,scale = 1, color = 'red'):

    plt.figure() #(figsize=(10, 10))
    plt.imshow (Image, cmap='gray')

    # Create a grid of coordinates (downsampled with step)
    y, x = np.mgrid[0:Image.shape[0]:step, 0:Image.shape[1]:step]

    # Downsample flow vectors for clarity
    u_small = u[::step, ::step]*scale
    v_small = v[::step, ::step]*scale

    # Draw quiver plot
    plt.quiver (x, y, u_small, v_small, color=color, angles='xy', scale_units='xy')

    plt.title ("Optical Flow (Horn-Schunck)")
    plt.axis ('off')
    plt.show ()



def computeDenseFlow(ref_frame, work_frame):
    # Ensure grayscale
    if len(ref_frame.shape) == 3:
        ref_frame = cv.cvtColor(ref_frame, cv.COLOR_BGR2GRAY)
    if len(work_frame.shape) == 3:
        work_frame = cv.cvtColor(work_frame, cv.COLOR_BGR2GRAY)

    # Calculate dense optical flow
    flow = cv.calcOpticalFlowFarneback(ref_frame, work_frame,
                                       None,
                                       pyr_scale=0.5,
                                       levels=5,
                                       winsize=15,
                                       iterations=5,
                                       poly_n=7,
                                       poly_sigma=1.5,
                                       flags=0)
    return flow  # shape: (H, W, 2), where [:,:,0] = flow_x, [:,:,1] = flow_y


def draw_DenseFlow(Image, Flow, step = 10, scale=1, color = 'red'):

    plt.figure()
    plt.imshow(Image, cmap='gray')

    # Create a grid of coordinates (downsampled with step)
    y, x = np.mgrid[0:Image.shape[0]:step, 0:Image.shape[1]:step]

    u = Flow[:,:,0]
    v = Flow[:,:,1]

    cv.cartToPolar(u,v)

    plt.title ("Optical Flow (Dense-Flow)")
    plt.axis ('off')
    plt.show ()



def horn_schunck_optical_flow(frame1, frame2, alpha=1.0, delta=1e-3, num_levels=3, max_iter=100):

    def warp_image(img, flow_u, flow_v):
        h, w = img.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow_u).astype(np.float32)
        map_y = (grid_y + flow_v).astype(np.float32)
        return cv.remap(img, map_x, map_y, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)

    def compute_gradients(img1, img2):
        fx = cv.Sobel(img1, cv.CV_64F, 1, 0, ksize=3) / 8.0 + cv.Sobel(img2, cv.CV_64F, 1, 0, ksize=3) / 8.0
        fy = cv.Sobel(img1, cv.CV_64F, 0, 1, ksize=3) / 8.0 + cv.Sobel(img2, cv.CV_64F, 0, 1, ksize=3) / 8.0
        ft = img2 - img1
        return fx, fy, ft

    def horn_schunck_single_level(img1, img2, alpha, delta, max_iter):
        fx, fy, ft = compute_gradients(img1, img2)
        u = np.zeros(pyramid1[0].shape, dtype=np.float64)
        v = np.zeros(pyramid1[0].shape, dtype=np.float64)


        kernel = np.array([[1/12, 1/6, 1/12],
                           [1/6,    0, 1/6],
                           [1/12, 1/6, 1/12]])

        for _ in range(max_iter):
            u_avg = cv.filter2D(u, -1, kernel)
            v_avg = cv.filter2D(v, -1, kernel)
            num = fx * u_avg + fy * v_avg + ft
            den = alpha**2 + fx**2 + fy**2
            du = -fx * num / den
            dv = -fy * num / den
            if np.max(np.abs(du)) < delta and np.max(np.abs(dv)) < delta:
                break
            u += du
            v += dv
        return u, v

    # # Convert images to grayscale float32
    # frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    pyramid1 = [frame1]
    pyramid2 = [frame2]
    for _ in range(1, num_levels):
        pyramid1.insert(0, cv.pyrDown(pyramid1[0]))
        pyramid2.insert(0, cv.pyrDown(pyramid2[0]))

    u = np.zeros_like(pyramid1[0])
    v = np.zeros_like(pyramid1[0])

    for level in range(num_levels):
        img1 = pyramid1[level]
        img2 = pyramid2[level]
        if level != 0:
            u = cv.resize(u * 2, (img1.shape[1], img1.shape[0]), interpolation=cv.INTER_LINEAR)
            v = cv.resize(v * 2, (img1.shape[1], img1.shape[0]), interpolation=cv.INTER_LINEAR)
            img1 = warp_image(img1, u, v)

        du, dv = horn_schunck_single_level(img1, img2, alpha, delta, max_iter)
        u += du
        v += dv

    return u, v
