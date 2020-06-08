import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_img(img, cmap=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap)
    plt.show()


rainbow = cv2.imread('../DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)
img = rainbow
print(img.shape)
mask = np.zeros(img.shape[:2], np.uint8)

# plt.imshow(mask, cmap='gray')
# plt.show()

mask[300:400, 100:400] = 255
# plt.imshow(mask, cmap='gray')
# plt.show()

masked_img = cv2.bitwise_and(img, img, mask=mask)
show_masked_img = cv2.bitwise_and(show_rainbow, show_rainbow, mask=mask)
# plt.imshow(masked_img, cmap='gray')
# plt.show()

# bgr
hist_mask_values_red = cv2.calcHist(
    [rainbow], channels=[2], mask=mask, histSize=[256], ranges=[0, 256])
hist_values_red = cv2.calcHist(
    [rainbow], channels=[2], mask=None, histSize=[256], ranges=[0, 256])
# plt.plot(hist_mask_values_red)
# plt.title('red histogram for masked rainbow')
# plt.show()

# plt.plot(hist_values_red)
# plt.title('red histogram for normal rainbow')
# plt.show()


gorilla = cv2.imread('../DATA/gorilla.jpg', 0)
# display_img(gorilla, 'gray')

hist_values = cv2.calcHist([gorilla], channels=[0],
                           mask=None, histSize=[256], ranges=[0, 256])
# plt.plot(hist_values)
# plt.show()

eq_gorilla = cv2.equalizeHist(gorilla)
# display_img(eq_gorilla, 'gray')

hist_values_1 = cv2.calcHist([eq_gorilla], channels=[0],
                             mask=None, histSize=[256], ranges=[0, 256])
# plt.plot(hist_values_1)
# plt.show()

color_gorilla = cv2.imread('../DATA/gorilla.jpg')
show_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2RGB)
# display_img(show_gorilla)
hsv = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2HSV)
hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
eq_color_gorilla = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
display_img(eq_color_gorilla)
