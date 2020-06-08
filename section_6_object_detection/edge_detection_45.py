import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../DATA/sammy_face.jpg')
# plt.imshow(img)
# plt.show()


# edges = cv2.Canny(image=img, threshold1=0, threshold2=255)
# plt.imshow(edges)
# plt.show()


med_val = np.median(img)
print(med_val)

# lower threshhold to either 0 or 70% of the median value whichever is greater
lower = int(max(0, 0.7*med_val))
# upper threshhold to either 0 or 130% of the max value = 255, whichever is smaller
upper = int(min(255, 1.3*med_val))

# edges = cv2.Canny(image=img, threshold1=lower, threshold2=upper)
# plt.imshow(edges)
# plt.show()

blurred_img = cv2.blur(img, ksize=(5, 5))
edges = cv2.Canny(image=blurred_img, threshold1=lower, threshold2=upper+50)
plt.imshow(edges)
plt.show()
