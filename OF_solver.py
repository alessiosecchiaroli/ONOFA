import numpy as np
import cv2 as cv
from scipy.ndimage.filters import convolve as filter2
import matplotlib.pyplot as plt

def get_derivatives(img1, img2):
    #derivative masks
    x_kernel = np.array([[-1, 1], [-1, 1]]) * 0.25
    y_kernel = np.array([[-1, -1], [1, 1]]) * 0.25
    t_kernel = np.ones((2, 2)) * 0.25

    fx = filter2(img1,x_kernel) + filter2(img2,x_kernel)
    fy = filter2(img1, y_kernel) + filter2(img2, y_kernel)
    ft = filter2(img1, -t_kernel) + filter2(img2, t_kernel)

    return [fx,fy, ft]


def computeHS(beforeImg, afterImg, alpha, delta):

    #removing noise
    beforeImg  = cv.GaussianBlur(beforeImg, (5,5), 0)
    afterImg = cv.GaussianBlur(afterImg, (5,5), 0)

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
        d = 4 * alpha**2 + fx**2 + fy**2
        prev = u

        u = u_avg - fx * (p / d)
        v = v_avg - fy * (p / d)

        diff = np.linalg.norm(u - prev, 2)
        #converges check (at most 300 iterations)
        if  diff < delta or iter_counter > 300:
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

def computeLK(ref_frame, work_frame,exp_max_disp):


    lk_params = dict (winSize=(15, 15),
                      maxLevel=6,
                      criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # initial displacement
    y_dim, x_dim = ref_frame.shape

    # init = np.random.normal(loc=x_dim/3,scale=exp_max_disp,size=x_dim)

    # Generate grid of points (e.g., 500 total)
    x_points = x_dim
    y_points = y_dim
    xs = np.random.normal (loc=exp_max_disp, scale=1, size=x_points)
    ys = np.random.uniform (0, 1, size=y_points)

    # Clip points to stay inside image bounds
    xs = np.clip (xs, 0, x_dim - 1)
    ys = np.clip (ys, 0, y_dim - 1)

    # Stack and reshape to (N, 1, 2), type float32
    p0 = np.vstack((xs, ys)).T.astype(np.float32).reshape(-1, 1, 2)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK (ref_frame, work_frame, p0, None, **lk_params)

    return p1,st,err


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

def draw_OF_LK(Image, u, v, step = 10,scale = 1, color = 'red'):

    plt.figure() #(figsize=(10, 10))
    plt.imshow (Image, cmap='gray')

    # Create a grid of coordinates (downsampled with step)
    y, x = np.mgrid[0:Image.shape[0]:step, 0:Image.shape[1]:step]

    # Downsample flow vectors for clarity
    u_small = u[::step]*scale
    v_small = v[::step]*scale

    # Draw quiver plot
    plt.quiver (x, y, u_small, v_small, color=color, angles='xy', scale_units='xy')

    plt.title ("Optical Flow (Lucas-Kanade)")
    plt.axis ('off')
    plt.show ()

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