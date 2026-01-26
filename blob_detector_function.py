import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def cross_finder_top(img, blr):

    img_to_work_on = img

    # The image is blurred to reduce noise and improve the detection of blobs
    # It's especially important to wash out the crosses
    # img_blur = cv.GaussianBlur(img_to_work_on, (blr, blr), 0)
    img_blur = cv.medianBlur(img_to_work_on, blr)

    # visualize the blurred image
    plt.imshow(img_blur,cmap='gray')
    plt.show()

    params = cv.SimpleBlobDetector_Params()

    # Filter by Color
    params.filterByColor = True
    params.minThreshold = 10
    params.maxThreshold = 255

    # Filter by Area
    params.filterByArea = True
    params.minArea = 2000
    params.maxArea = 8000

    # Filter by Circularity
    params.filterByCircularity = False  
    params.minCircularity = 0.4
    # params.maxCircularity = 1.0

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.5
    params.maxConvexity = 1.0

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    params.maxInertiaRatio = 1.0


    detector = cv.SimpleBlobDetector_create(params)

    # detect blobs  
    keypoints = detector.detect(img_blur)

    # Draw detected blobs as red circles
    img_with_keypoints = cv.drawKeypoints(
        img, keypoints, np.array([]),
        (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Show keypoints
    plt.imshow(img_with_keypoints)
    plt.show()

    return keypoints

def cross_finder(img, blr):

    img_to_work_on = img

    # The image is blurred to reduce noise and improve the detection of blobs
    # It's especially important to wash out the crosses
    # img_blur = cv.GaussianBlur(img_to_work_on, (blr, blr), 0)
    img_blur = cv.medianBlur(img_to_work_on, blr)

    # visualize the blurred image
    # plt.imshow(img_blur,cmap='gray')
    # plt.show()

    params = cv.SimpleBlobDetector_Params()

    # Filter by Color
    params.filterByColor = True
    params.minThreshold = 10
    params.maxThreshold = 255

    # Filter by Area
    params.filterByArea = True
    params.minArea = 2000
    params.maxArea = 8000

    # Filter by Circularity
    params.filterByCircularity = False  
    params.minCircularity = 0.4
    # params.maxCircularity = 1.0

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.5
    params.maxConvexity = 1.0

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.5
    params.maxInertiaRatio = 1.0


    detector = cv.SimpleBlobDetector_create(params)

    # detect blobs  
    keypoints = detector.detect(img_blur)
    position = [kp.pt for kp in keypoints]  # List of (x, y) tuples
    size = [kp.size for kp in keypoints]  # List of sizes

    # # Draw detected blobs as red circles
    # img_with_keypoints = cv.drawKeypoints(
    #     img, keypoints, np.array([]),
    #     (255, 255, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # )

    # # Show keypoints
    # plt.imshow(img_with_keypoints)
    # plt.show()

    return position, size

def dots_finder(img):

    img_to_work_on = img

    # The image is blurred to reduce noise and improve the detection of blobs
    # It's especially important to wash out the crosses
    # img_blur = cv.GaussianBlur(img_to_work_on, (blr, blr), 0)
    img_blur = cv.medianBlur(img_to_work_on, blr)

    # visualize the blurred image
    plt.imshow(img_blur,cmap='gray')
    plt.show()

    params = cv.SimpleBlobDetector_Params()

    # Filter by Color
    params.filterByColor = True
    params.minThreshold = 0
    params.maxThreshold = 150

    # Filter by Area
    params.filterByArea = True
    params.minArea = 110
    params.maxArea = 260

    # Filter by Circularity
    params.filterByCircularity = False  
    params.minCircularity = 0.4
    # params.maxCircularity = 1.0

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.5
    params.maxConvexity = 1.0

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.5
    params.maxInertiaRatio = 1.0


    detector = cv.SimpleBlobDetector_create(params)

    # detect blobs  
    keypoints = detector.detect(img_blur)

    # Draw detected blobs as red circles
    img_with_keypoints = cv.drawKeypoints(
        img, keypoints, np.array([]),
        (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Show keypoints
    plt.imshow(img_with_keypoints)
    plt.show()

    return keypoints

