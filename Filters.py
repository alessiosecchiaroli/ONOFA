import os
import cv2 as cv
import numpy as np

def bandpass_filter(image,low_cutoff, high_cutoff):

    # Get image dimensions
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Perform FFT
    dft = np.fft.fft2 (image)
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
    # plt.show ()

    return cv.normalize(img_filtered, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

def Blur_subtraction(img,blur_size):

    blurred_img = cv.GaussianBlur(img,(blur_size,blur_size),0)

    subtracted_img = img-blurred_img

    return subtracted_img