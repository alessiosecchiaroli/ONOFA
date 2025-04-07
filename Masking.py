import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def mask_points(img):

    # Show image
    plt.imshow (img, cmap='gray')
    plt.title ("Click to define points (up to 100). close the window to finish early.")
    plt.axis ('on')

    # ginput with timeout and maximum of 100 points
    points_raw = plt.ginput (n=100, timeout=0)  # timeout=0 means wait indefinitely

    # Separate x and y coordinates
    x_coords = [p[0] for p in points_raw]
    y_coords = [p[1] for p in points_raw]

    print (f"Total points selected: {len (points_raw)}")
    print ("X coordinates:", x_coords)
    print ("Y coordinates:", y_coords)

    # adjust the data format
    points = np.array(points_raw,np.int32)
    pts = points.reshape((-1,1,2))

    # draw the polygon
    mask = cv.polylines (img, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

    # masked_frame = cv.bitwise_and (img, img, mask=mask)

    return mask