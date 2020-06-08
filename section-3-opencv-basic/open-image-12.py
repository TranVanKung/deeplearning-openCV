import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../DATA/00-puppy.jpg')
# print(type(img))
print(img.shape)

# matplotlib -> red, green, blue
# opencv -> blue, green, red
# plt.imshow(img)
# plt.show()

fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(fix_img.shape)
# plt.imshow(fix_img)
# plt.show()

img_gray = cv2.imread('../DATA/00-puppy.jpg', cv2.IMREAD_GRAYSCALE)
print(img_gray.shape)
print(img_gray.max())
print(img_gray.min())
# plt.imshow(img_gray, cmap='gray')
# plt.show()

new_img = cv2.resize(fix_img, (1000, 400))
print(type(new_img))
# plt.imshow(new_img)
# plt.show()

w_ratio = 0.5
h_ratio = 0.5

new_img_1 = cv2.resize(fix_img, (0, 0), fix_img, w_ratio, h_ratio)
print(new_img_1.shape)
# plt.imshow(new_img_1)
# plt.show()

new_img_2 = cv2.flip(fix_img, -1)
# plt.imshow(new_img_2)
# plt.show()

cv2.imwrite('total_new_img.jpg', fix_img)

fig = plt.figure(figsize=(2, 2))
ax = fig.add_subplot(111)
ax.imshow(fix_img)
plt.show()
