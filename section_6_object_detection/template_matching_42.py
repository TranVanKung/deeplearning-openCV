import cv2
import numpy as np
import matplotlib.pyplot as plt

full = cv2.imread('../DATA/sammy.jpg')
full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
# plt.imshow(full)
# plt.show()

face = cv2.imread('../DATA/sammy_face.jpg')
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

# plt.imshow(face)
# plt.show()

# một trick hay trong python, tự tìm xem có hàm nào matching với string hay không, nếu có thì trả lại hàm đó
my_string = 'sum'
print(eval(my_string))

my_func = eval(my_string)
print(my_func([1, 2, 3]))

# all the 6 methods for comparison in a list
# note how we are using strings, later on we'll use the eval() function to convert the function
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCORR_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for m in methods:
    # create a copy
    full_copy = full.copy()
    method = eval(m)

    # template matching
    result = cv2.matchTemplate(full_copy, face, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if method in [cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF]:
        top_left = min_loc  # (x, y)
    else:
        top_left = max_loc  # (x, y)

    height, width, channels = face.shape
    bottom_right = (top_left[0]+width, top_left[1]+height)
    cv2.rectangle(full_copy, top_left, bottom_right,
                  color=(255, 0, 0), thickness=10)
    #   plt and show the image
    plt.subplot(121)
    plt.imshow(result, cmap='gray')
    plt.title('heatmap of template matching')

    plt.subplot(122)
    plt.imshow(full_copy)
    plt.title('detection of template')
    # title with the method used
    plt.suptitle(m)
    plt.show()
    print('/n')

# my_method = eval('cv2.TM_CCOEFF')
# res = cv2.matchTemplate(full, face, my_method)
# plt.imshow(res)
# plt.show()
