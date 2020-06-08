import cv2
import numpy as np
import matplotlib.pyplot as plt

blank_img = np.zeros(shape=(512, 512, 3), dtype=np.int16)
print(blank_img.shape)

# plt.imshow(blank_img)
# plt.show()

cv2.rectangle(blank_img, pt1=(381, 10), pt2=(
    500, 80), color=(0, 155, 0), thickness=10)
# plt.imshow(blank_img)
# plt.show()

cv2.rectangle(blank_img, pt1=(200, 200), pt2=(
    300, 300), color=(0, 0, 255), thickness=10)
# plt.imshow(blank_img)
# plt.show()

cv2.circle(img=blank_img, center=(100, 100),
           radius=50, color=(255, 0, 0), thickness=8)
# plt.imshow(blank_img)
# plt.show()

cv2.circle(img=blank_img, center=(400, 400),
           radius=50, color=(255, 0, 0), thickness=-1)
# plt.imshow(blank_img)
# plt.show()

cv2.line(blank_img, pt1=(0, 0), pt2=(512, 512),
         color=(102, 255, 244), thickness=5)
# plt.imshow(blank_img)
# plt.show()

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(blank_img, text="Hello", org=(
    10, 500), fontFace=font, fontScale=4, color=(255, 255, 255), thickness=3, lineType=cv2.LINE_AA)
# plt.imshow(blank_img)
# plt.show()

blank_img_1 = np.zeros(shape=(512, 512, 3), dtype=np.int32)
# plt.imshow(blank_img_1)
# plt.show()

vertices = np.array([[100, 300], [200, 200], [400, 300],
                     [200, 400]], dtype=np.int32)
print(vertices.shape)
points = vertices.reshape((-1, 1, 2))
print(points.shape)
cv2.polylines(blank_img_1, [points], isClosed=True,
              color=(255, 0, 0), thickness=5)
plt.imshow(blank_img_1)
plt.show()
