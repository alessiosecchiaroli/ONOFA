import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

from Old_studd.Functions import  tellme

# find the picture
root = os.getcwd()
imgPath = os.path.join(root, '../Exp_pics/220C_reference.bmp')
img = cv.imread(imgPath)

#rescale it
height, width, _ = img.shape
scale = 1/1

heightScale = int(height*scale)
widthScale = int(width*scale)

# equalize the histogram to improve contrast
img = cv.resize(img,(widthScale,heightScale),interpolation=cv.INTER_LINEAR)
img = cv.cvtColor (img, cv.COLOR_BGR2GRAY)
img = cv.equalizeHist (img)

plt.imshow (img,cmap='gray')
plt.show()
#
# xc1 = input('x position of top cross: ', )
# xc2 = input('x position of bottom cross: ', )
# yc1 = input('y position of top cross: ', )
# yc2 = input('y position of bottom cross: ', )

plt.title("Click on the top cross, then the bottom cross")
# plt.show()

# # Let the user click on the image to get coordinates
# points = plt.ginput(2, timeout=-1)  # Capture two clicks
#
# # Extract x and y positions from the clicked points
# (xc1, yc1), (xc2, yc2) = points

tellme('You will select the two crosses, starting from the top one, click to begin')

plt.waitforbuttonpress()

while True:
    pts = []
    while len(pts) < 2:
        tellme('Select the 2 points with mouse')
        pts = np.asarray(plt.ginput(2, timeout=-1))
        if len(pts) < 2:
            tellme('Too few points, starting over')
            time.sleep(1)  # Wait a second

    ph = plt.fill(pts[:, 0], pts[:, 1], 'r', lw=2)

    tellme('Happy? Key click for yes, mouse click for no')

    if plt.waitforbuttonpress():
        break

    # Get rid of fill
    for p in ph:
        p.remove()
print(pts)

#
center = np.array([round(np.mean(pts[:,0])), round(np.mean(pts[:,1]))])
#
# # the span has been hardcoded
# # based on a single frame
x_span = 100 #px
y_span = 300 #px

size = img.shape
mask = np.zeros([size[0], size[1]],dtype='uint8')
mask = cv.rectangle(mask,(center[0]-x_span,center[1]-y_span),(center[0]+x_span, center[1]+y_span),(255,255,255),-1)


z = cv.bitwise_and(img,img,mask=mask)

plt.figure()
plt.imshow(z)
plt.show()
#
# canny_edge = cv.Canny(z,255,255)
#
# plt.subplot(121)
# plt.imshow (canny_edge)
#
# distResol = 1
# angleResol = np.pi/180
# threshold = 150
# lines = cv.HoughLines(canny_edge,distResol,angleResol,threshold)
#
# k = 3000
#
# for curLine in lines:
#
#
#     rho,theta = curLine[0]
#     dhat = np.array([np.cos(theta)],[np.sin(theta)])
#     d = rho*dhat
#     lhat = np.array([-np.sin(theta)],[np.cos(theta)])
#     p1 = d + k*lhat
#     p2 = d- k*lhat
#     p1 = p1.astype(int)
#     p2 = p2.astype(int)
#
#     cv.line(z,(p1[0][0],p1[1][0]), (p2[0][0],p2[1][0]),(255,255,255),10)
#
# plt.subplot(122)
# plt.imshow (z)
z1 = cv.GaussianBlur(z,(3,3),3)

orb = cv.ORB_create()
keypoints = orb.detect(z1,None)
keypoints,_ = orb.compute(z1,keypoints)
z1 = cv.drawKeypoints(z1,keypoints,z1,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure()
plt.imshow(z1)
plt.show()