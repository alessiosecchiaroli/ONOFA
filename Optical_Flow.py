import cv2 as cv
import numpy as np

def detect_dots(frame_gray):
    # Set up the SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()
    # params.filterByColor = True
    # params.blobColor = 255
    params.filterByArea = True
    params.minArea = 15
    params.maxArea = 60
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False

    # Create a detector with the parameters
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(frame_gray)

    # Convert to the format LK expects: (N,1,2)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)

    return points

def computeLK_tracking(ref_gray, work_gray):
    # Detect points
    p0 = detect_dots(ref_gray)

    lk_params = dict(winSize=(21, 21),
                     maxLevel=6,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1))

    # Run LK Optical Flow
    p1, st, err = cv.calcOpticalFlowPyrLK(ref_gray, work_gray, p0, None, **lk_params)

    # Filter valid points
    p0_good = p0[st >= 0.25]
    p1_good = p1[st >= 0.25]

    return p0_good, p1_good, err

def draw_tracking(ref_frame, p0, p1):
    vis = cv.cvtColor (ref_frame.copy (), cv.COLOR_GRAY2BGR)
    for pt0, pt1 in zip(p0, p1):
        x0, y0 = pt0.ravel().astype(int)
        x1, y1 = pt1.ravel().astype(int)
        cv.line(vis, (x0, y0), (x1, y1), (0, 0, 255), 1)
        cv.circle(vis, (x1, y1), 3, (0, 0, 255), 1)
    return vis

def compute_my_HS(ref_frame,work_frame,alpha):

    def get_derivative(img, ksize=5):

        # use Sobel filter for derivatives
        # default kernel = 3 px

        # x derivative
        # 64f is more accurate and find more edges (I don't know exactly why, it's stated from openCV)
        dx_64f = cv.Sobel(img,cv.CV_64F,1,0,ksize=3)
        dx_abs = np.absolute(dx_64f)
        dx_8u = np.uint8(dx_abs)

        # y derivative
        dy_64f = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)
        dy_abs = np.absolute(dy_64f)
        dy_8u = np.uint8(dy_abs)

        return dx_8u, dy_8u

        # Lagrangian multiplier
        # needed for brightness correction
        alpha = 0.5

        # obtaining the spatial derivatives using the function
        dIdx_img1, dIdy_img1 = get_derivative(ref_frame)
        dIdx_img2, dIdy_img2 = get_derivative(work_frame)

        # obtaining temporal derivative by subtracting the image
        dIdT = work_frame - ref_frame

        # initialize u and v with zeros of same size and format as grad_T
        u = np.ones_like(grad_T)
        v = np.ones_like(u)

        def double_Integral(xmin, xmax, ymin, ymax, nx, ny, A):
            dS = ((xmax - xmin) / (nx - 1)) * ((ymax - ymin) / (ny - 1))

            A_Internal = A[1:-1, 1:-1]

            # sides: up, down, left, right
            (A_u, A_d, A_l, A_r) = (A[0, 1:-1], A[-1, 1:-1], A[1:-1, 0], A[1:-1, -1])

            # corners
            (A_ul, A_ur, A_dl, A_dr) = (A[0, 0], A[0, -1], A[-1, 0], A[-1, -1])

            return dS * (np.sum (A_Internal)
                         + 0.5 * (np.sum (A_u) + np.sum (A_d) + np.sum (A_l) + np.sum (A_r))
                         + 0.25 * (A_ul + A_ur + A_dl + A_dr))

        # grad_dx = dIdx_img2 - dIdx_img1
        # grad_dy = dIdy_img2 - dIdy_img1

        fun_to_min = (dIdx_img1 * u + dIdy_img1 * v + dIdT) + alpha * (np.abs(u)**2 + np.abs(v)**2 )





