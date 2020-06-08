import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../Data/rainbow.jpg', 0)
# plt.imshow(img, cmap='gray')
# plt.show()

print(img.max())
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# plt.imshow(thresh1, cmap='gray')
# plt.show()

img = cv2.imread('../DATA/crossword.jpg', 0)
# plt.imshow(img, cmap='gray')
# plt.show()


def show_pic(img):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


# show_pic(img)
ret1, th1 = cv2.threshold(img, 200, 155, cv2.THRESH_BINARY)
# show_pic(th1)

th2 = cv2.adaptiveThreshold(
    img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
# show_pic(th2)

blended = cv2.addWeighted(src1=th1, alpha=0.6, src2=th2, beta=0.4, gamma=0)
show_pic(blended)
