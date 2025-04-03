import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def openImage():
    root = os.getcwd ()
    imgPath = os.path.join (root, '../Exp_pics/220C_reference.bmp')
    img = cv.imread (imgPath)

    height, width, _ = img.shape
    scale = 1 / 5

    heightScale = int (height * scale)
    widthScale = int (width * scale)

    img = cv.resize (img, (widthScale, heightScale), interpolation=cv.INTER_LINEAR)

    cv.imshow('img',img)
    cv.waitKey(0)


def callback(input):
    pass

def cannyEdge():
    root = os.getcwd()
    imgPath = os.path.join(root, '../Exp_pics/220C_reference.bmp')
    img = cv.imread(imgPath)

    height, width, _ = img.shape
    scale = 1/1

    heightScale = int(height*scale)
    widthScale = int(width*scale)

    img = cv.resize(img,(widthScale,heightScale),interpolation=cv.INTER_LINEAR)
    img = cv.cvtColor (img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist (img)

    plt.imshow(img)
    plt.show()

    img = cv.GaussianBlur(img,(3,3),3)
    winname = 'canny'
    cv.namedWindow(winname)
    cv.createTrackbar('minThres',winname,0,255,callback)
    cv.createTrackbar('maxThres',winname,0,255,callback)

    while True:
        if cv.waitKey(1) == ord('q'):
            break

        minThres = cv.getTrackbarPos('minThres',winname)
        maxThres = cv.getTrackbarPos('maxThres',winname)
        cannyEdge = cv.Canny(img,minThres,maxThres)
        cv.imshow(winname,cannyEdge)

    cv.destroyAllWindows()

def lowpass_filter():
    root = os.getcwd ()
    imgPath = os.path.join (root, '../Exp_pics/220C_reference.bmp')
    img = cv.imread (imgPath)

    height, width, _ = img.shape
    scale = 1 / 5

    heightScale = int (height * scale)
    widthScale = int (width * scale)

    img = cv.resize (img, (widthScale, heightScale), interpolation=cv.INTER_LINEAR)
    img = cv.cvtColor (img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist (img)

    plt.subplot (231)
    plt.imshow (img, cmap='gray')

    imgDFT = cv.dft(np.float32(img),flags=cv.DFT_COMPLEX_OUTPUT)
    imgDFT_dB = 20*np.log(cv.magnitude(imgDFT[:,:,0],imgDFT[:,:,1]))

    plt.subplot(232)
    plt.imshow(imgDFT_dB,cmap='gray')

    imgDFTshift = np.fft.fftshift(imgDFT)
    imgDFTshift_dB = 20*np.log(cv.magnitude(imgDFTshift[:,:,0],imgDFTshift[:,:,1]))

    plt.subplot (233)
    plt.imshow (imgDFTshift_dB, cmap='gray')

    r,c = img.shape #rows and columns
    mask = np.zeros((r,c,2),np.uint8) # apply mask,depth 2 because the image DFT have reals and imaginary layers
    offset = 100
    mask[int(r/2)-offset:int(r/2)+offset, int(c/2)-offset:int(c/2)+offset] = 1
    plt.subplot(234)
    plt.imshow(mask[:,:,0])

    imgDFTshift_LP = imgDFTshift*mask
    imgDFTshift_LP_dB = 20*np.log(cv.magnitude(imgDFTshift_LP[:,:,0],imgDFTshift_LP[:,:,1]))

    plt.subplot(235)
    plt.imshow(imgDFTshift_LP_dB,cmap='gray')

    imgInvDFT_LP = np.fft.ifftshift(imgDFTshift_LP)
    imgDFT_LP = cv.idft(imgInvDFT_LP)
    img_LP = cv.magnitude(imgDFT_LP[:,:,0],imgDFT_LP[:,:,1])

    plt.subplot(236)
    plt.imshow(img_LP,cmap='gray')

    plt.show()

def LP_FILT():
    root = os.getcwd ()
    imgPath = os.path.join (root, '../Exp_pics/220C_4bar.bmp')
    img = cv.imread (imgPath)

    # img = cv.imread(imgPath,1)
    # img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    height, width, _ = img.shape
    scale = 1 / 5

    heightScale = int (height * scale)
    widthScale = int (width * scale)

    img = cv.resize (img, (widthScale, heightScale), interpolation=cv.INTER_LINEAR)
    img = cv.cvtColor (img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist (img)

    # %% Filtering.
    # Construct a Gaussian Kernel of size nb x nb
    n_b = 111  # it should be an odd number
    x_a = np.linspace (-(n_b - 1) / 2, (n_b - 1) / 2, n_b)
    xg, yg = np.meshgrid (x_a, x_a)
    sigma = 50
    w = 1 / (2 * np.pi * sigma ** 2) * np.exp (-(xg ** 2 + yg ** 2) / (2 * sigma ** 2))
    plt.pcolor (xg, yg, w)

    G = signal.convolve2d (img, w, boundary='symm', mode='same')
    # Get the spectra of the filtered image
    G_HAT_ABS = np.abs (np.fft.fftshift (np.fft.fft2 (G - np.mean (G))))

    ig, ax = plt.subplots (figsize=(4, 4))
    plt.pcolor (img - G)  # We use a log transform for visibility
    ax.set_aspect ('equal')  # Set equal aspect ratio
    plt.gca ().invert_yaxis ()
    ax.set_xlabel ('$x[p]$', fontsize=14)
    ax.set_ylabel ('$y[p]$', fontsize=14)
    plt.colorbar ()
    plt.tight_layout (pad=1, w_pad=0.5, h_pad=1.0)
    plt.tight_layout ()

    plt.show()


def bandpass_filter(low_cutoff, high_cutoff):
    # Convert to grayscale if necessary
    root = os.getcwd ()
    imgPath = os.path.join (root, '../Exp_pics/220C_4bar.bmp')
    img = cv.imread (imgPath)

    # img = cv.imread(imgPath,1)
    # img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    height, width, _ = img.shape
    scale = 1 / 5

    heightScale = int (height * scale)
    widthScale = int (width * scale)

    img = cv.resize (img, (widthScale, heightScale), interpolation=cv.INTER_LINEAR)
    img = cv.cvtColor (img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist (img)

    # if len (image.shape) == 3:
    #     image = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Perform FFT
    dft = np.fft.fft2 (img)
    dft_shift = np.fft.fftshift (dft)

    # Create bandpass filter mask
    mask = np.zeros ((rows, cols), np.uint8)
    for i in range (rows):
        for j in range (cols):
            distance = np.sqrt ((i - crow) ** 2 + (j - ccol) ** 2)
            if low_cutoff < distance < high_cutoff:
                mask[i, j] = 1

    # Apply mask
    filtered_dft = dft_shift * mask

    # Perform inverse FFT
    dft_ishift = np.fft.ifftshift (filtered_dft)
    img_filtered = np.fft.ifft2 (dft_ishift)
    img_filtered = np.abs (img_filtered)

    # plt.imshow (img_filtered, cmap='gray')
    #
    # plt.show ()

    # return cv2.normalize (img_filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

def HoughLine():
    root = os.getcwd ()
    imgPath = os.path.join (root, '../Exp_pics/220C_4bar.bmp')
    img = cv.imread (imgPath)

    # img = cv.imread(imgPath,1)
    # img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    height, width, _ = img.shape
    scale = 1 / 4

    heightScale = int (height * scale)
    widthScale = int (width * scale)

    img = cv.resize (img, (widthScale, heightScale), interpolation=cv.INTER_LINEAR)
    img = cv.cvtColor (img, cv.COLOR_BGR2GRAY)
    img = cv.equalizeHist (img)

    imgBlur = cv.GaussianBlur(img,(3,3),3)
    # canny_edge = cv.Canny(imgBlur,100,160)
    canny_edge = cv.Canny(imgBlur,255,255)

    plt.figure()
    plt.subplot(141)
    plt.imshow(img)
    plt.subplot(142)
    plt.imshow(imgBlur)
    plt.subplot(143)
    plt.imshow(canny_edge)


    distResol = 1
    angleResol = np.pi/180
    threshold = 150
    lines = cv.HoughLines(canny_edge,distResol,angleResol,threshold)

    k = 3000

    for curLine in lines:


        rho,theta = curLine[0]
        dhat = np.array([np.cos(theta)],[np.sin(theta)])
        d = rho*dhat
        lhat = np.array([-np.sin(theta)],[np.cos(theta)])
        p1 = d + k*lhat
        p2 = d- k*lhat
        p1 = p1.astype(int)
        p2 = p2.astype(int)

        cv.line(img,(p1[0][0],p1[1][0]), (p2[0][0],p2[1][0]),(255,255,255),10)

    plt.subplot(144)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    # cannyEdge()
    # openImage()
    # lowpass_filter()
    # LP_FILT()
    bandpass_filter(5,150)
    # HoughLine()