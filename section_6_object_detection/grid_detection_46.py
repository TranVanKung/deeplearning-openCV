import cv2
import numpy as np
import matplotlib.pyplot as plt

flat_chess = cv2.imread('../DATA/flat_chessboard.png')
# plt.imshow(flat_chess)
# plt.show()


found, corners = cv2.findChessboardCorners(flat_chess, (7, 7))

print(found, corners)
cv2.drawChessboardCorners(flat_chess, (7, 7), corners, found)
# plt.imshow(flat_chess)
# plt.show()

dots = cv2.imread('../DATA/dot_grid.png')
# plt.imshow(dots)
# plt.show()
found, corners = cv2.findCirclesGrid(
    dots, (10, 10), cv2.CALIB_CB_SYMMETRIC_GRID)
print(found)
cv2.drawChessboardCorners(dots, (10, 10), corners, found)
plt.imshow(dots)
plt.show()
