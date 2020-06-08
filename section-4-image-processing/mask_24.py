import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('../Data/dog_backpack.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('../Data/watermark_no_copy.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img2, (600, 600))

print(img1.shape)

x_offset = img1.shape[1] - 600
y_offset = img1.shape[0] - 600

rows, cols, channels = img2.shape

region = img1[y_offset:img1.shape[0], x_offset:img1.shape[1]]
img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
# plt.imshow(img2gray, cmap='gray')
# plt.show()

mask_inv = cv2.bitwise_not(img2gray)
# plt.imshow(mask_inv, cmap='gray')
# plt.show()
print(mask_inv.shape)

white_background = np.full(img2.shape, 255, dtype=np.uint8)
# plt.imshow(white_background)
# plt.show()
print(white_background.shape)

bk = cv2.bitwise_or(white_background, white_background, mask=mask_inv)
print(bk.shape)
# plt.imshow(bk)
# plt.show()

foreground = cv2.bitwise_or(img2, img2, mask=mask_inv)
# plt.imshow(foreground)
# plt.show()

final_region = cv2.bitwise_or(region, foreground)
# plt.imshow(final_region)
# plt.show()

large_image = img1
small_image = final_region
large_image[y_offset:y_offset+small_image.shape[0],
            x_offset:x_offset+small_image.shape[1]] = small_image
plt.imshow(large_image)
plt.show()
