import cv2
import numpy as np

# Create a black image with white circles (blobs)
img = np.zeros((400, 400), dtype=np.uint8)
cv2.circle(img, (100, 100), 20, (255, 255, 255), -1)
cv2.circle(img, (300, 100), 40, (255, 255, 255), -1)
cv2.circle(img, (200, 300), 30, (255, 255, 255), -1)

cv2.imshow('visual', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Set up SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200

# Filter by area
params.filterByArea = True
params.minArea = 100
params.maxArea = 5000

# Filter by circularity
params.filterByCircularity = True
params.minCircularity = 0.7

# Filter by convexity
params.filterByConvexity = False

# Filter by inertia
params.filterByInertia = False

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(img)

# Draw detected blobs as red circles
img_with_keypoints = cv2.drawKeypoints(
    img, keypoints, np.array([]),
    (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Show the image
cv2.imshow("Blobs Detected", img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
