import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

road = cv2.imread('../DATA/road_image.jpg')
road_copy = np.copy(road)
# plt.imshow(road)
# plt.show()

# print(road.shape)
marker_image = np.zeros(road.shape[:2], dtype=np.int32)
segments = np.zeros(road.shape, dtype=np.uint8)
# print(cm.tab10(0))


def create_rgb(i):
    return tuple(np.array(cm.tab10(i)[:3])*255)


colors = []
for i in range(10):
    colors.append(create_rgb(i))
# print(colors)

# global variable
# color choice
n_markers = 10  # 0-9
current_marker = 1
# marker updated by watershed
marks_updated = False


def mouse_callback(event, x, y, flags, param):
    global marks_updated

    if event == cv2.EVENT_LBUTTONDOWN:
        # markers passed to the watershed algo
        cv2.circle(marker_image, (x, y), 10, (current_marker), -1)

        # user see on the road image
        cv2.circle(road_copy, (x, y), 10, colors[current_marker], -1)
        marks_updated = True


cv2.namedWindow('Road image')
cv2.setMouseCallback('Road image', mouse_callback)

while True:
    cv2.imshow('Watershed segment', segments)
    cv2.imshow('Road image', road_copy)

    # close all window
    k = cv2.waitKey(1)
    if k == 27:
        break
    # clear all the colors press C key
    elif k == ord('c'):
        road_copy = road.copy()
        marker_image = np.zeros(road.shape[:2], dtype=np.int32)
        segments = np.zeros(road.shape, dtype=np.uint8)
    # update color choice
    elif k > 0 and chr(k).isdigit():
        current_marker = int(chr(k))
    # update the markings
    if marks_updated:
        marker_image_copy = marker_image.copy()
        cv2.watershed(road, marker_image_copy)
        segments = np.zeros(road.shape, dtype=np.uint8)
        for color_ind in range(n_markers):
            segments[marker_image_copy == (color_ind)] = colors[color_ind]

cv2.destroyAllWindows()
