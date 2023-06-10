import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('aadc.jpeg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

img_prewittx = cv2.filter2D(img_gray, -1, kernelx)
img_prewitty = cv2.filter2D(img_gray, -1, kernely)
img_prewitt = img_prewittx + img_prewitty

fig, axes = plt.subplots(4, 2, figsize=(20, 20))
ax = axes.ravel()

ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0].set_title("Citra Input")
ax[1].hist(img_gray.ravel(), bins=256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_prewittx, cmap='gray')
ax[2].set_title("Citra Prewitt X")
ax[3].hist(img_prewittx.ravel(), bins=256)
ax[3].set_title("Histogram Citra Prewitt X")

ax[4].imshow(img_prewitty, cmap='gray')
ax[4].set_title("Citra Prewitt Y")
ax[5].hist(img_prewitty.ravel(), bins=256)
ax[5].set_title("Histogram Citra Prewitt Y")

ax[6].imshow(img_prewitt, cmap='gray')
ax[6].set_title("Citra Prewitt")
ax[7].hist(img_prewitt.ravel(), bins=256)
ax[7].set_title("Histogram Citra Prewitt")

fig.tight_layout()
plt.show()