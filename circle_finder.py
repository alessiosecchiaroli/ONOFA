import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from Filters import bandpass_filter
from Filters import Blur_subtraction

def circles_finder(img,blur_lvl,xmin,xmax,r_t,r_w):

    # save the image in another image, that will be use to write on
    img_for_circles = img
    # img_blur = img_for_circles
    # # THESE ARE A LIST OF FILTERS, good luck finding the right one
    # img_blur = cv.bilateralFilter(img_for_circles, blur_lvl, 200, 200) # Bilateral, is non linear and good for edges
    # img_blur = cv.filter2D(img_for_circles, -1, np.ones((blur_lvl, blur_lvl), np.float32) / (blur_lvl ** 2)) # this is a simple box filter (it uses a convolution)
    # img_blur = cv.medianBlur(img_for_circles, blur_lvl) # this is a median filter, it is good for removing noise
    img_blur = cv.GaussianBlur(img_for_circles, (blur_lvl, blur_lvl), 0) # this is a gaussian filter, it is good for removing noise

    # img_blur = bandpass_filter(img_for_circles, 19, 300) # this is a bandpass filter, it is good for edge highlighting
    # img_blur = Blur_subtraction(img_for_circles, blur_lvl) # this is a blur subtraction filter, it is good for edge highlighting

    # plt.subplot(1,2,1)
    # plt.imshow(img_for_circles,cmap='gray')
    # plt.subplot(1,2,2)
    # plt.imshow(img_blur,cmap='gray')
    # plt.title('blurred image')
    # plt.show()

    # find edges with canny
    # cannyEdge_visual(img_blur) # just to have a look and pick the limits
    # edges = cv.Canny (img_blur, 75, 150)

    # apply Hough Circle transform
    # the parameter are tuned on my specific case (kinda hardcoded)
    # param1 and param2 are the two thresholds for the Canny edge detector
    # param1 is the higher one, param2 is the lower one
    # circles = cv.HoughCircles (
    #     img_blur, cv.HOUGH_GRADIENT, dp=1.2, minDist=300,
    #     param1=100, param2=40, minRadius=40, maxRadius=60)
    
    # This one uses Scharr's method
    # param1 is the threshold for derivative, param2 is the perfectness of the circle
    circles = cv.HoughCircles (
    img_blur, cv.HOUGH_GRADIENT_ALT, dp=1.5, minDist=300,
    param1=300, param2=0.7, minRadius=40, maxRadius=60)

    if circles is not None:

        circles = np.uint16 (np.around (circles))
        # # This commented part write all the circles in the picture
        # # It is better to skip it and just write the two circles we're interested into
        # for i in circles[0, :]:
        #     center = (i[0], i[1])
        #     cv.circle(img_for_circles, center, 1, (0, 255, 0), 3)  # Draw center in green
        #     radius = i[2]
        #     cv.circle(img_for_circles, center, radius, (255, 0, 255), 3)  # Draw outline in magenta
        
        # plt.imshow(img_for_circles,cmap='gray')
        # plt.show()

    # Define filtering conditions
        x_min, x_max = xmin, xmax # hardocded based on the picture
        radius_target, radius_window = r_t, r_w  # Accept radii in the range x+-y (e.g. 50 +- 3 pixels)

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
    
        plt.imshow(img_for_circles,cmap='gray')
        plt.title('detected circles')
        plt.show()

        print(f"Filtered circles (x in [{x_min}, {x_max}], r in [{radius_target - radius_window}, {radius_target + radius_window}]): {len(filtered_circles)}")

        return selected_circles

