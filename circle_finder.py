import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def circles_finder(img,blur_lvl,xmin,xmax):

    # save the image in another image, that will be use to write on
    img_for_circles = img

    # slightly blur the image to remove noise
    # the value of the kernel (11) is hardcoded
    img_blur = cv.medianBlur(img_for_circles, blur_lvl)

    # find edges with canny
    # cannyEdge_visual(img_blur) # just to have a look and pick the limits
    # edges = cv.Canny (img_blur, 75, 150)

    # apply Hough Circle transform
    # the parameter are tuned on my specific case (kinda hardcoded)
    circles = cv.HoughCircles (
        img_blur, cv.HOUGH_GRADIENT, dp=1, minDist=300,
        param1=100, param2=30, minRadius=20, maxRadius=100)


    if circles is not None:
        circles = np.uint16 (np.around (circles))
        # # This commented part write all the circles in the picture
        # # It is better to skip it and just write the two circles we're interested into
        # for i in circles[0, :]:
        #     center = (i[0], i[1])
        #     cv.circle(img_for_circles, center, 1, (0, 255, 0), 3)  # Draw center in green
        #     radius = i[2]
        #     cv.circle(img_for_circles, center, radius, (255, 0, 255), 3)  # Draw outline in magenta

    # Define filtering conditions
        x_min, x_max = xmin, xmax # hardocded based on the picture
        radius_target, radius_window = 52, 9  # Accept radii in the range 46-56

    # Filter detected circles
        filtered_circles = [
            circle for circle in circles[0, :]
            if
            x_min <= circle[0] <= x_max and (radius_target - radius_window) <= circle[2] <= (radius_target + radius_window)
        ]

        if len (filtered_circles) >= 2:
            selected_circles = sorted (filtered_circles, key=lambda c: c[1])[:2]
        else:
            selected_circles = filtered_circles  # If fewer than 2, just return available ones

        for c in selected_circles:
            center = (c[0], c[1])
            cv.circle (img_for_circles, center, 1, (0, 255, 0), 3)  # Green dot at center
            cv.circle (img_for_circles, center, c[2], (255, 0, 255), 3)  # Magenta outline

        # plt.imshow(img_for_circles)
        # plt.show()

        return selected_circles