import cv2

img = cv2.imread('../DATA/00-puppy.jpg')

while True:
    cv2.imshow('Puppy', img)

    # if we've waited at least 1 ms and we've pressed the Esc
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
