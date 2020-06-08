import cv2
import numpy as np
import matplotlib.pyplot as plt

# original bgr open cv
dark_horse = cv2.imread('../DATA/horse.jpg')
# converted to rgb for show
show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB)

rainbow = cv2.imread('../DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)

blue_bricks = cv2.imread('../DATA/bricks.jpg')
show_bricks = cv2.cvtColor(blue_bricks, cv2.COLOR_BGR2RGB)

# plt.imshow(show_horse)
# plt.show()

# plt.imshow(show_rainbow)
# plt.show()

# plt.imshow(show_bricks)
# plt.show()

# open cv bgr
hist_values = cv2.calcHist([dark_horse], channels=[
                           0], mask=None, histSize=[256], ranges=[0, 256])
# print(hist_values.shape)
# plt.plot(hist_values)
# plt.show()

img = dark_horse
color = ('b', 'g', 'r')

for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
    plt.ylim([0, 500000])
plt.title('Histogram')
plt.show()
