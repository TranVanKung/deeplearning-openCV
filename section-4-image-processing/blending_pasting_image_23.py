import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('../Data/dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('../Data/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# plt.imshow(img1)
# plt.show()
# plt.imshow(img2)
# plt.show()

# blending the image of the same size
img1 = cv2.resize(img1, (1200, 1200))
img2 = cv2.resize(img2, (1200, 1200))

# plt.imshow(img2)
# plt.show()
blended = cv2.addWeighted(src1=img1, alpha=0.5, src2=img2, beta=0.5, gamma=0)

# plt.imshow(blended)
# plt.show()

# overlap small image on top of a larger image (no blending)
# numpy reassignment
img3 = cv2.imread('../Data/dog_backpack.png')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img4 = cv2.imread('../Data/watermark_no_copy.png')
img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)

img4 = cv2.resize(img4, (600, 600))
# plt.imshow(img4)
# plt.show()

# plt.imshow(img3)
# plt.show()

large_img = img3
small_img = img4

x_offset = 0
y_offset = 0
x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]
large_img[y_offset:y_end, x_offset: x_end] = small_img

# plt.imshow(large_img)
# plt.show()
