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







