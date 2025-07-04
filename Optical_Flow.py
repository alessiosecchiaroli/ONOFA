import cv2 as cv
import numpy as np
from scipy.ndimage import convolve

def detect_dots(frame_gray):
    # Set up the SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 25
    params.maxArea = 50
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
                     maxLevel=5,
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



def lucas_kanade_optical_flow(img1, img2, window_size=5):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    img1 = cv.GaussianBlur(img1,(5,5),0)
    img2 = cv.GaussianBlur(img2,(5,5),0)

    # Compute gradients
    kernel_x = np.array([[-1, 1], [-1, 1]]) * 0.25
    kernel_y = np.array([[-1, -1], [1, 1]]) * 0.25
    kernel_t = np.ones((2, 2)) * 0.25

    Ix = convolve(img1, kernel_x) + convolve(img2, kernel_x)
    Iy = convolve(img1, kernel_y) + convolve(img2, kernel_y)
    It = convolve(img2, kernel_t) - convolve(img1, kernel_t)

    u = np.zeros(img1.shape)
    v = np.zeros(img1.shape)

    half_w = window_size // 2

    for y in range(half_w, img1.shape[0] - half_w):
        for x in range(half_w, img1.shape[1] - half_w):
            Ix_win = Ix[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1].flatten()
            Iy_win = Iy[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1].flatten()
            It_win = It[y - half_w:y + half_w + 1, x - half_w:x + half_w + 1].flatten()

            A = np.stack((Ix_win, Iy_win), axis=1)
            b = -It_win

            # Least squares solution to A @ [u, v] = b
            ATA = A.T @ A
            if np.linalg.det(ATA) >= 1e-4:  # ensure it's invertible
                uv = np.linalg.inv(ATA) @ A.T @ b
                u[y, x] = uv[0]
                v[y, x] = uv[1]

    return u, v




