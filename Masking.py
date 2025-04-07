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
    np.save("220C",pts)

    return mask

def shape_isolation(image, shape_points):

    # Create a blank single-channel mask the same size as the image
    mask = np.zeros (image.shape[:2], dtype=np.uint8)

    # Fill the polygon (instead of just drawing the outline)
    cv.fillPoly (mask, [shape_points], 255)

    # Optional: visualize the filled area
    # cv.imshow ('Mask', mask);
    # cv.waitKey (0)

    # Apply the mask to keep only the area inside the polygon
    image_Mask = cv.bitwise_and (image, image, mask=mask)
    # work_img_M = cv.bitwise_and (work_img, work_img, mask=mask)

    if len (image.shape) == 2:
        # Grayscale image: just use a scalar
        background = np.full_like (image, 255)  # or 0 for black
    else:
        # Color image: need to broadcast (255, 255, 255) to image shape
        background = np.full (image.shape, background_color, dtype=np.uint8)

    # Combine the masked image with the background
    image_final = np.where (mask == 255, image, background)
    # work_img_final = np.where (mask == 255, work_img, background)

    return image_final