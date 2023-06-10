import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('aadc.jpeg')
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_sobelx = cv.Sobel(img_gray, cv.CV_8U, 1, 0, ksize=5)
img_sobely = cv.Sobel(img_gray, cv.CV_8U, 0, 1, ksize=5)
img_sobel = img_sobelx + img_sobely

fig, axes = plt.subplots(4, 2, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
ax[0].set_title("Citra Input")
ax[1].hist(img_gray.ravel(), bins=256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_sobelx, cmap='gray')
ax[2].set_title("Citra Sobel X")
ax[3].hist(img_sobelx.ravel(), bins=256)
ax[3].set_title("Histogram Citra Sobel X")

ax[4].imshow(img_sobely, cmap='gray')
ax[4].set_title("Citra Sobel Y")
ax[5].hist(img_sobely.ravel(), bins=256)
ax[5].set_title("Histogram Citra Sobel Y")

ax[6].imshow(img_sobel, cmap='gray')
ax[6].set_title("Citra Sobel")
ax[7].hist(img_sobel.ravel(), bins=256)
ax[7].set_title("Histogram Citra Sobel")

fig.tight_layout()
plt.show()